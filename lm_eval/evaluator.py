import json
import os
import warnings

import torch
from tqdm import tqdm

from lm_eval import tasks
from lm_eval.generation import parallel_generations
from lm_eval.utils import calculate_entropy, calculate_spike_entropy
from watermark.watermark import WatermarkDetector
from watermark.sweet import SweetDetector
from watermark.ie import IEDetector
from watermark.ewd import EWDDetector
from utils.model import Unified_Feature_Translator
from utils.utils import save_json_to_file, load_json_from_file
import pdb

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator:
    def __init__(self, accelerator,  tokenizer, args, model=None, embed_model=None, embed_tokenizer=None, classifier=None, proxy_model=None, proxy_tokenizer=None, max_seq_len=512, batch_size=32):

        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path
        if args.entropy_file_path:
            print(f'Entropy file path: {args.entropy_file_path}')
            dir_path, base_path = os.path.split(args.metric_output_path)
            base_path = base_path.replace('.json', f'perturbed_{self.args.entropy_acc}.json')
            self.metric_output_path = os.path.join(dir_path, base_path)
            self.entropy = load_json_from_file(args.entropy_file_path)

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

        if args.ie:
            self.embed_model = embed_model
            self.embed_tokenizer = embed_tokenizer
            self.classifier = classifier
            self.feature_extractor = Unified_Feature_Translator(self.tokenizer, self.embed_tokenizer, self.embed_model)
            self.max_seq_len = max_seq_len
            self.batch_size = batch_size
        
        elif args.sweet_proxy:
            self.proxy_model = proxy_model
            self.proxy_tokenizer = proxy_tokenizer

    def generate_text(self, task_name, start_index=None, end_index=None):
        task = tasks.get_task(task_name)(start_index=start_index, end_index=end_index)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        if self.args.ie:
            generations = parallel_generations(
                task,
                dataset,
                self.accelerator,
                self.model,
                self.tokenizer,
                n_tasks=n_tasks,
                args=self.args,
                classifier=self.classifier,
                embed_model=self.embed_model,
                embed_tokenizer=self.embed_tokenizer
            )
        else:
            generations = parallel_generations(
                task,
                dataset,
                self.accelerator,
                self.model,
                self.tokenizer,
                n_tasks=n_tasks,
                args=self.args
            )
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def watermark_detect(self, task_name, generations, watermark_detector):
        task = tasks.get_task(task_name)()
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(generations)

        def tokenize(example):
            inputs = self.tokenizer(
                example,
                padding=True,
                truncation=True,
                return_tensors="pt",
                #max_length=args.max_length_generation,
            )
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }

        # for results saving
        result = {"z_threshold": self.args.detection_z_threshold}
        detect_list = []
        ent_list = []
        ent_map = {}
        detection_results = []
        len_list = []

        prompt_contents = [task.get_prompt(dataset[sample]) for sample in range(n_tasks)]

        if self.accelerator.is_main_process:
            for idx, gens in tqdm(enumerate(generations), total=len(generations)):
                n_detection = 0
                for idx2, gen in enumerate(gens):

                    # we don't check all n_samples generations
                    if n_detection >= self.args.n_detection:
                        continue

                    prefix = prompt_contents[idx]
                    tokenized_prefix = tokenize(prefix)['input_ids'].squeeze()
                    prefix_len = len(tokenized_prefix)

                    # if the prompt is not part of generation
                    try:
                        assert gen.startswith(prefix)
                    except AssertionError:
                        print(f"{idx}, {idx2}")
                        pdb.set_trace()

                    tokenized_text = tokenize(gen)['input_ids'].squeeze()

                    # if tokenized are not same
                    try:
                        assert torch.equal(tokenized_text[:prefix_len],tokenized_prefix), "Tokenized prefix must be a prefix of the tokenized text"
                    except AssertionError:
                        print(f"{idx}, {idx2}")
                        # tokenizing issue.. check at least the lens are same
                        if len(tokenized_text[:prefix_len]) == len(tokenized_prefix):
                            pass
                        else:
                            pdb.set_trace()

                    # if len of generation is 0, check next genertion
                    if len(tokenized_text) - prefix_len == 0:
                        continue
                    else:
                        if idx2 != 0:
                            print(idx2)
                        n_detection += 1

                    # entropy calculation
                    if self.args.sweet:
                        self.model = self.model.to(self.accelerator.device)
                        entropy = calculate_entropy(self.model, tokenized_text.to(self.accelerator.device))
                        # we need to shift entropy to the right, so the first item is dummy
                        entropy = [0] + entropy[:-1]
                        # entropy [0, entropy_id_1, entropy_id_2, ...]
                        ent_map[idx] = entropy
                        ent_list += entropy[prefix_len:]

                    elif self.args.ie:
                        entropy = []
                        for batch_feature in self.feature_extractor.feature_extractor_for_bert(tokenized_text, self.max_seq_len, self.batch_size):
                            batch_feature = batch_feature.float().to(self.accelerator.device)
                            with torch.no_grad():
                                scores = torch.softmax(self.classifier(batch_feature), dim=1)
                            entropy += scores[:, 1].tolist()
                        entropy = [1.0] + entropy[:-1]
                        ent_map[idx] = entropy
                        ent_list += entropy[prefix_len:]

                    # WLLM detector
                    if self.args.wllm:
                        detection_result = watermark_detector.detect( # no batch
                            tokenized_text=tokenized_text,
                            tokenized_prefix=tokenized_prefix,
                        )
                        if not detection_result.pop('invalid', False):
                            detect_list.append(1 if detection_result['prediction'] else 0)
                            detection_results.append(detection_result)

                    # SWEET detector
                    elif self.args.sweet:
                        detection_result = watermark_detector.detect( # no batch
                            tokenized_text=tokenized_text,
                            tokenized_prefix=tokenized_prefix,
                            entropy=entropy,
                        )
                        if not detection_result.pop('invalid', False):
                            detect_list.append(1 if detection_result['prediction'] else 0)
                            detection_results.append(detection_result)

                    elif self.args.ie:
                        detection_result = watermark_detector.detect( # no batch
                            tokenized_text=tokenized_text,
                            tokenized_prefix=tokenized_prefix,
                            entropy=entropy,
                            weighted_z_score=self.args.weighted_z_score
                        )
                        if not detection_result.pop('invalid', False):
                            detect_list.append(1 if detection_result['prediction'] else 0)
                            detection_results.append(detection_result)

                    # general info
                    len_list.append(len(tokenized_text) - prefix_len)

                # all generations' len are 0
                if n_detection < self.args.n_detection:
                    print(f"all {idx}th generations are 0 len.")
                    len_list.append(0)

            if detect_list:
                print(f"total samples: {len(detect_list)}, positive samples: {sum(detect_list)}")
                result["total_samples"] = len(detect_list)
                result["positive_samples"] = sum(detect_list)
                result["detection_rate"] = sum(detect_list) / len(detect_list)

            if len_list:
                print(f"average len of generated code : {sum(len_list) / len(len_list)}")
                result["mean_len"] = sum(len_list) / len(len_list)

            if detection_results:
                # watermarking_fraction
                if "watermarking_fraction" in detection_results[0]:
                    wfs = [d['watermarking_fraction'] for d in detection_results]
                    result["watermarking_fraction"] = sum(wfs) / len(wfs)

                if "green_fraction" in detection_results[0]:
                    gfs = [d['green_fraction'] for d in detection_results]
                    result["green_fraction"] = sum(gfs) / len(gfs)

                result["raw_detection_results"] = detection_results

        return result

    def evaluate(self, task_name):
        task = tasks.get_task(task_name)()
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        # detecting generated code
        if not self.args.detect_human_code:
            generations, references = self.generate_text(task_name)
        # detecting human code
        else:
            task_dataset = task.get_dataset()
            generations = []
            for i in range(len(task_dataset)):
                full_human = task.get_full_data(task_dataset[i])

                # APPS dataset doesn't always have solution
                if full_human:
                    generations.append([full_human])

            references = [task.get_reference(task_dataset[i]) for i in range(len(task_dataset))]

        # load watermark detector
        watermark_detector = None

        if self.args.wllm:
            watermark_detector = WatermarkDetector(vocab=list(self.tokenizer.get_vocab().values()),
                                        gamma=self.args.gamma,
                                        tokenizer=self.tokenizer,
                                        z_threshold=self.args.detection_z_threshold)
        
        elif self.args.sweet:
            watermark_detector = SweetDetector(vocab=list(self.tokenizer.get_vocab().values()),
                                        gamma=self.args.gamma,
                                        tokenizer=self.tokenizer,
                                        z_threshold=self.args.detection_z_threshold,
                                        entropy_threshold=self.args.entropy_threshold)
            
        elif self.args.ie:
            watermark_detector = IEDetector(vocab=list(self.tokenizer.get_vocab().values()),
                                        gamma=self.args.gamma,
                                        tokenizer=self.tokenizer,
                                        z_threshold=self.args.detection_z_threshold,
                                        entropy_threshold=self.args.entropy_threshold)

        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                if self.args.save_generations:
                    with open(self.args.save_generations_path, "w") as fp:
                        json.dump(generations, fp)
                        print(
                            f"generations were saved at {self.args.save_generations_path}"
                        )
                if self.args.save_references:
                    with open("references.json", "w") as fp:
                        json.dump(references, fp)
                        print("references were saved at references.json")
            
            watermark_detection_results = self.watermark_detect(task_name, generations, watermark_detector)

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"

            print("Evaluating generations...")
            results, pass_info = task.process_results(generations, references)

            results["watermark_detection"] = watermark_detection_results

            results["pass_info"] = pass_info

            return results