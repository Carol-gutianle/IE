import pdb
import torch

from utils.model import Unified_Feature_Translator
from watermark.watermark import WatermarkDetector, WatermarkLogitsProcessor

class IELogitsProcessor(WatermarkLogitsProcessor):
    
    def __init__(self, origin_tokenizer, embed_tokenizer, embed_model, classifier, *args, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = Unified_Feature_Translator(origin_tokenizer, embed_tokenizer, embed_model)
        self.classifier = classifier
        self.classifier.to(self.device)
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids, scores):
        if self.rng is None:
            self.rng = torch.Generator()
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]
        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids
        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)
        features = self.feature_extractor.feature_extractor_next_token(input_ids, max_seq_len=512).to(self.device)
        with torch.no_grad():
            entropy = torch.softmax(self.classifier(features.float()), dim=1)[:, 1]
        entropy_mask = (entropy < 0.5).view(-1, 1).to(green_tokens_mask.device)
        green_tokens_mask = green_tokens_mask * entropy_mask
        scores = self._bias_greenlist_logits(
            scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta
        )
        return scores
    
class IEDetector(WatermarkDetector):
    
    def __init__(self, *args, tokenizer, z_threshold = 4, ignore_repeated_bigrams = False, **kwargs):
        super().__init__(*args, tokenizer=tokenizer, z_threshold=z_threshold, ignore_repeated_bigrams=ignore_repeated_bigrams, **kwargs)

    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        prefix_len: int,
        entropy: list[float],
        weighted_z_score: bool = False,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_watermarking_fraction: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
        return_color_tokens: bool = False,
        return_generate_tokens: bool = False
    ):
        score_dict = dict()
        prefix_len = max(self.min_prefix_len, prefix_len)

        if self.ignore_repeated_bigrams:
            raise NotImplementedError("not implemented for entropy")

        num_tokens_generated = len(input_ids) - prefix_len
        if num_tokens_generated < 1:
            print(f"only {num_tokens_generated} generated : cannot score.")
            score_dict["invalid"] = True
            return score_dict

        try:
            assert len(entropy) == len(input_ids)
        except AssertionError:
            print("len(entropy) != len(input_ids)")
            import pdb
            pdb.set_trace()

        entropy_wo_prefix = entropy[prefix_len:]
        num_tokens_scored = num_tokens_generated - sum([True if ent > 0.5 else False for ent in entropy_wo_prefix])
        entropy_pass = []
        for ent in entropy_wo_prefix:
            if ent < 0.5:
                entropy_pass.append(ent)

        if num_tokens_scored < 1:
            assert num_tokens_scored == 0
            # regarding as human generated
            return {
                "num_tokens_generated": num_tokens_generated,
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "watermarking_fraction": 0,
                "green_fraction": 0,
                "z_score": -100.0,
                "p_value": 1,
            }
        
        green_token_count, green_token_mask = 0, []
        # green_token: 0, red_token: 1, grey_token: 2
        color_tokens = []
        generated_tokens = []
        for idx in range(prefix_len, len(input_ids)):
            curr_token = input_ids[idx]
            generated_tokens.append(curr_token.item())
            greenlist_ids = self._get_greenlist_ids(input_ids[:idx])
            try:
                if entropy[idx] < 0.5:
                    if curr_token in greenlist_ids:
                        green_token_count += 1
                        green_token_mask.append(True)
                        color_tokens.append(0)
                    else:
                        green_token_mask.append(False)
                        color_tokens.append(1)
                else:
                    # when entropy is low; i.e., watermarking is not applied
                    green_token_mask.append(False)
                    color_tokens.append(2)
            except:
                import pdb
                pdb.set_trace()
        
        if green_token_count > num_tokens_scored:
            import pdb 
            pdb.set_trace()

        score_dict.update(dict(num_tokens_generated=num_tokens_generated))
        if return_generate_tokens:
            score_dict.update(dict(generated_tokens=generated_tokens))
        if return_color_tokens:
            score_dict.update(dict(color_tokens=color_tokens))
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_watermarking_fraction:
            score_dict.update(
                dict(watermarking_fraction=(num_tokens_scored / num_tokens_generated))
            )
        if return_green_fraction:
            score_dict.update(
                dict(green_fraction=(green_token_count / num_tokens_scored))
            )
        if return_z_score:
            if weighted_z_score:
                z_score = self._compute_weighted_z_score(green_token_count, num_tokens_scored, entropy_pass)
            else:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(z_score=z_score))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict