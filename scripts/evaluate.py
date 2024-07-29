
import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizer
from utils import get_device, tokenize_data, NewsDataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(model_path, data_path):
    # Load Model and Tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    dataset = load_dataset('csv', data_files={'validation': data_path})
    tokenized_datasets = dataset.map(lambda x: tokenize_data(tokenizer, x, max_length=128), batched=True)

    # DataLoader
    eval_dataset = NewsDataset(tokenized_datasets['validation'])
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False)

    # Evaluation Function
    def evaluate(model, data_loader, device):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        return accuracy, precision, recall, f1

    # Evaluate
    accuracy, precision, recall, f1 = evaluate(model, eval_loader, device)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing validation data')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
