🧠 BERT Sentiment Classifier – Fine-Tuning

Fine-tuning BERT (bert-base-uncased) for binary sentiment analysis using Hugging Face’s transformers and datasets libraries.

🚀 Project Overview

This notebook demonstrates how to fine-tune a pre-trained BERT model on a binary sentiment dataset (e.g., positive vs. negative reviews).
The workflow includes:

Dataset loading and train/validation/test split

Tokenization and preprocessing

Model fine-tuning using Trainer API

Evaluation on validation and test sets

🧩 Requirements

Install the necessary dependencies before running the notebook:

pip install torch transformers datasets evaluate scikit-learn

📂 Dataset Preparation

Use your dataset (e.g., capterra_reviews_binary.csv) with a text and label column.
Example split code:

from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('capterra_reviews_binary.csv')
train, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

train.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False)
test.to_csv('test.csv', index=False)

⚙️ Fine-Tuning Steps

Load tokenizer and model

from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


Tokenize the dataset

from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'val.csv', 'test': 'test.csv'})
def tokenize_fn(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True)
tokenized_datasets = dataset.map(tokenize_fn, batched=True)


Train the model

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer
)

trainer.train()

📊 Results

After training for 5 epochs, the model achieved:

Split	Accuracy	Loss	ROC-AUC
Validation	94.3%	0.228	0.548
Test	94.0%	0.239	0.580

(Metrics computed using the Hugging Face evaluate library.)

🧪 Evaluation

Run the evaluation step after training:

trainer.evaluate(tokenized_datasets['test'])


Expected output:

{'eval_loss': 0.2386,
 'eval_accuracy': 0.9400,
 'eval_roc_auc': 0.5802,
 'epoch': 5.0}

💾 Saving and Loading the Model
trainer.save_model('./bert-sentiment-finetuned')
tokenizer.save_pretrained('./bert-sentiment-finetuned')


Load later for inference:

from transformers import pipeline
sentiment = pipeline('text-classification', model='./bert-sentiment-finetuned')
sentiment("This app is fantastic!")

🧠 Key Learnings

Fine-tuning BERT yields high accuracy even on small datasets.

ROC-AUC may vary depending on dataset balance and label quality.

You can improve performance by:

Increasing dataset size

Using class weighting

Applying data augmentation (e.g., back-translation)

Fine-tuning more epochs with smaller learning rate

📈 Example Output
Epoch 5/5 complete.
Validation accuracy: 94.3%
Test accuracy: 94.0%
Model ready for inference!

🏁 Credits

Pretrained model: BERT-base-uncased

Libraries: transformers, datasets, evaluate, scikit-learn, torch

Author: Vishnu Sunil (thelightweilder)
