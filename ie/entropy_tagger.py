# Tokenwise entropy tagger
import os
import datetime
import wandb
import pickle
import hydra
import torch
import random
from tqdm import tqdm
from rich import print
from copy import deepcopy
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial


from utils.utils import set_seed
from utils.model import Classifier
from utils.dataset import CustomIterableDataset, TrainValTestIterableDataset

    
def collate_fn(batch, threshold):
    hidden_states = []
    labels = []
    for example in batch:
        hidden_states.append(example['hidden_states'])
        if example['label'] > threshold:
            labels.append(0)
        else:
            labels.append(1)
    labels = [float(label) for label in labels]
    return {
        'hidden_states': torch.stack(hidden_states),
        'labels': torch.tensor(labels)
    }

def duplicate(tagger_dataset):
    seen_text = set()
    filtered_data = []
    for sample in tqdm(tagger_dataset):
        if sample['hidden_states'] not in seen_text:
            seen_text.add(sample['hidden_states'])
            filtered_data.append(sample)
    return filtered_data


def test(test_dataset_path, ckpt_path, collate_fn):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    classifier = Classifier(input_dim = 768).to(device)
    classifier.load_state_dict(torch.load(ckpt_path, weights_only=True))
    acc = evaluate(test_dataset_path, device, classifier, collate_fn)
    return acc

def evaluate(evaluate_dataset_path, device, model, collate_fn):
    with open(evaluate_dataset_path, 'rb') as f:
        tagger_dataset = pickle.load(f)
    
    tagger_dataset = TrainValTestIterableDataset(tagger_dataset)
    tagger_dataset = CustomIterableDataset(tagger_dataset)
    tagger_dataloader = DataLoader(tagger_dataset, batch_size=1, collate_fn=collate_fn)

    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch in tqdm(tagger_dataloader, leave=False):
            hidden_states, labels = batch['hidden_states'], batch['labels']
            hidden_states = hidden_states.to(device)
            labels = labels.to(device)
            outputs = model(hidden_states)
            predicted = torch.argmax(outputs, dim=1)
            for pred, label in zip(predicted, labels):
                if pred == label:
                    correct += 1
                total += 1
    return correct / total


@hydra.main(version_base=None, config_path="config", config_name="tagger")
def main(args):

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_model_path = os.path.join("MODEL_DIR", f"tagger_model_{args.dataset_name}_{curr_time}_entropy{args.entropy_threshold}.pt")

    wandb.init(project="ie", config={
        "dataset_name": args.dataset_name,
        "train_dataset_path": args.train_dataset_path,
        "evaluate_dataset_path": args.evaluate_dataset_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "save_model_path": save_model_path
    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open(args.train_dataset_path, 'rb') as f:
        tagger_dataset = pickle.load(f)

    tagger_dataset = duplicate(tagger_dataset)

    print(f'Total number of samples: {len(tagger_dataset)}')
    print(f'Number of positive samples: {len([sample for sample in tagger_dataset if sample["label"] < args.entropy_threshold])}')
    print(f'Number of negative samples: {len([sample for sample in tagger_dataset if sample["label"] >= args.entropy_threshold])}')
    print('--' * 20)
    
    collate_fn_w_threshold = partial(collate_fn, threshold=args.entropy_threshold)

    tagger_dataset = TrainValTestIterableDataset(tagger_dataset)
    tagger_dataset = CustomIterableDataset(tagger_dataset)
    tagger_dataloader = DataLoader(tagger_dataset, batch_size=args.batch_size, collate_fn=collate_fn_w_threshold)


    classifier = Classifier(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()

    best_acc = evaluate(args.evaluate_dataset_path, device, classifier, collate_fn_w_threshold)
    print(f'Initial Evaluation Accuracy: {best_acc}')
    best_ckpt = None

    # train classifier model
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        with tqdm(enumerate(tagger_dataloader)) as inner_pbar:
            for step, batch in inner_pbar:
                optimizer.zero_grad()
                hidden_states, labels = batch['hidden_states'], batch['labels']
                hidden_states = hidden_states.to(device)
                labels = labels.to(device)
                outputs = classifier(hidden_states).squeeze(1)
                loss = criterion(outputs, labels.long())
                epoch_loss += loss.item()
                inner_pbar.set_description(f'Loss: {loss.item():.4f}, Step: {step}, Epoch: {epoch}')
                loss.backward()
                optimizer.step()
        
        # save the model with the highest accuracy
        acc = evaluate(args.evaluate_dataset_path, device, classifier, collate_fn_w_threshold)
        wandb.log({"accuracy": acc, "epoch": epoch})
        print(f'Evaluation Accuracy: {acc}')

        if acc > best_acc:
            print(f'Update model, best accuracy: {acc}!')
            best_acc = acc
            best_ckpt = deepcopy(classifier.state_dict())

        epoch_loss /= step + 1

        print(f'Epoch {epoch}, Mean Loss: {epoch_loss}, Step: {step + 1}')

        if epoch_loss  < args.early_stopping_threshold:
            print(f'Early stopping at epoch {epoch}, mean loss: {epoch_loss}')
            break

        print('*' * 20)

    torch.save(best_ckpt, save_model_path)
    print(f'Save model to {save_model_path}')
    final_evaluation_acc = evaluate(args.evaluate_dataset_path, device, classifier, collate_fn_w_threshold)
    print(f'Final Evaluation Accuracy: {final_evaluation_acc}')
    wandb.log({"final_evaluation_accuracy": final_evaluation_acc})
    final_test_acc = evaluate(args.test_dataset_path, device, classifier, collate_fn_w_threshold)
    print(f'Final Test Accuracy: {final_test_acc}')
    other_test_acc = evaluate(args.other_test_dataset_path, device, classifier, collate_fn_w_threshold)
    print(f'Other Test Accuracy: {other_test_acc}')
    wandb.log({"other_test_accuracy": other_test_acc})
    wandb.log({"final_test_accuracy": final_test_acc})
    wandb.finish()

if __name__ == "__main__":
    set_seed(42)
    main()