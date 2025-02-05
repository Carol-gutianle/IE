import os
import json
import pickle
import hydra
import torch
import random
import itertools
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import IterableDataset
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM 
from utils.model import Unified_Feature_Translator 
from sentence_transformers import SentenceTransformer

tagger_dataset_save_dir = ''

def load_json_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f'Error: The file {file_path} has not been found!')
    except json.JSONDecodeError:
        print(f'Error: Wrong json file!')

def calculate_entropy(model, tokenized_text):
    with torch.no_grad():
        output = model(tokenized_text, return_dict=True)
        probs = torch.softmax(output.logits, dim=-1)
        entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
        return [0] + entropy[0].cpu().tolist()[:-1]

class TrainValTestIterableDataset(IterableDataset):
    def __init__(self, data_source, seed=42):
        self.data_source = data_source
        self.total_size = len(data_source)
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            random.seed(self.seed)
        data_iter = iter(self.data_source)
        return itertools.islice(data_iter, 0, self.total_size)

class CustomIterableDataset(IterableDataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __iter__(self):
        for sample in self.hf_dataset:
            yield sample

def load_hf_dataset(dataset_name, split):
    text_col = None
    label_col = None
    if dataset_name == 'humaneval':
        dataset = load_dataset('openai_humaneval', split=split)
        text_col = 'prompt'
        label_col = 'canonical_solution'
    elif dataset_name == 'mbpp':
        dataset = load_dataset('mbpp', split=split)
        text_col = 'text'
        label_col = 'code'
    elif dataset_name == 'ds1000':
        dataset = load_dataset('xlangai/DS-1000', split=split)
        text_col = 'prompt'
        label_col = 'reference_code'
    else:
        raise NotImplementedError(f'Current dataset {dataset_name} is not implemented!')
    return dataset, text_col, label_col

def load_hf_dataset_w_generations(dataset_name, json_file_path):
    text_col = None
    label_col = 'generation'
    if dataset_name == 'humaneval':
        dataset = load_dataset('openai_humaneval', split='test')
        text_col = 'prompt'
    elif dataset_name == 'mbpp':
        dataset = load_dataset('mbpp', split='validation')
        text_col = 'text'
    elif dataset_name == 'ds1000':
        dataset = load_dataset('xlangai/DS-1000', split='test')
    else:
        raise  NotImplementedError(f'Current dataset {dataset_name} is not implemented!')
    json_data = []
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    def add_generation_column(example, idx):
        example['generation'] = json_data[idx][0]
        return example
    dataset = dataset.map(add_generation_column, with_indices=True)
    return dataset, text_col, label_col
    
class TaggerDataset:
    def __init__(self, dataset_name, split, embedding_model, origin_model, embedding_tokenizer, origin_tokenizer, threshold, prefix=True):
        def get_dataset():
            dataset, text_col, label_col = load_hf_dataset(dataset_name, split)
            train_dataset = TrainValTestIterableDataset(dataset)
            train_dataset = CustomIterableDataset(train_dataset)
            return train_dataset, text_col, label_col
        
        self.dataset_name = dataset_name
        self.dataset, self.text_col, self.label_col = get_dataset()
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.max_seq_len = self.embedding_tokenizer.model_max_length
        self.origin_model = origin_model
        self.origin_tokenizer = origin_tokenizer
        self.prefix = prefix
        self.threshold = threshold
        self.split = split
        self.feature_extractor = Unified_Feature_Translator(self.origin_tokenizer, self.embedding_tokenizer, self.embedding_model)

        self.origin_model.eval()
        self.embedding_model.eval()
    
    def prepare_data(self, batch_size=32):
        tagger_dataset = []
        total_tokens = 0
        dataset_pbar = tqdm(self.dataset, leave=False)
        for sample in dataset_pbar:
            if self.prefix:
                prompt = sample[self.text_col] + ' ' + sample[self.label_col]
            else:
                prompt = sample[self.label_col]
            input_ids = self.origin_tokenizer(prompt, return_tensors='pt')['input_ids'][0].to(self.origin_model.device)
            entropy_list = calculate_entropy(self.origin_model, input_ids.unsqueeze(0))
            num_tokens = len(input_ids)
            total_tokens += num_tokens
            dataset_pbar.set_postfix({'Entropy': sum(entropy_list[1:]) / len(entropy_list[1:])})
            idx = 0
            for batch_feature in self.feature_extractor.feature_extractor_for_bert(input_ids, self.max_seq_len, batch_size):
                num_batch = batch_feature.shape[0]
                for offset in range(num_batch):
                    if idx * batch_size + offset == 0:
                        continue
                    entropy = entropy_list[idx * batch_size + offset]
                    tagger_dataset.append({
                        'hidden_states': batch_feature[offset],
                        'label': entropy
                    })
                idx += 1    
        if self.prefix:
            save_path = os.path.join(tagger_dataset_save_dir, f'regressor_sentence_tagger_dataset_{self.dataset_name}_entropy{self.threshold}_{self.split}.pkl')
        else:
            save_path = os.path.join(tagger_dataset_save_dir, f'regressor_sentence_tagger_dataset_{self.dataset_name}_entropy{self.threshold}_{self.split}_wo_prefix.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(tagger_dataset, f)
        print(f'Save tagger_dataset to {save_path}')
        print(f'Total number of tokens: {total_tokens}')
        self.explore_dataset()

    def explore_dataset(self):
        if self.prefix:
            save_path = os.path.join(tagger_dataset_save_dir, f'regressor_sentence_tagger_dataset_{self.dataset_name}_entropy{self.threshold}_{self.split}.pkl')
        else:
            save_path = os.path.join(tagger_dataset_save_dir, f'regressor_sentence_tagger_dataset_{self.dataset_name}_entropy{self.threshold}_{self.split}_wo_prefix.pkl')
        with open(save_path, 'rb') as f:
            tagger_dataset = pickle.load(f)
        print(f'Current Path: {save_path}')
        print(f'Total number of samples: {len(tagger_dataset)}')
        print(f'Number of positive samples: {len([sample for sample in tagger_dataset if sample["label"]])}')
        print(f'Number of negative samples: {len([sample for sample in tagger_dataset if not sample["label"]])}')
        print(f'First sample: {tagger_dataset[0]}')
        print(f'Last sample: {tagger_dataset[-1]}')

@hydra.main(config_path="../config", config_name="dataset")
def run(args):                        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_model = AutoModel.from_pretrained(args.embedding_model_path, device_map={'':device})
    origin_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map={'':device})

    embedding_model.eval()
    origin_model.eval()

    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
    origin_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dataset = TaggerDataset(args.dataset_name, args.split, embedding_model, origin_model, embedding_tokenizer, origin_tokenizer, args.threshold)
    dataset.explore_dataset()

if __name__ == "__main__":
    run()