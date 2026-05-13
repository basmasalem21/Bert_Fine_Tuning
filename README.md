📘 BERT Fine-Tuning for Customer Sentiment Analysis

This project demonstrates how to fine-tune a BERT (Bidirectional Encoder Representations from Transformers) model for multi-class customer sentiment classification using the dataset:

👉 JosefGoldstein/aimlessinnovations_customer_sentiment_v2

📊 Dataset

The dataset contains customer reviews labeled with sentiment classes:

very_negative
negative
neutral
positive
very_positive
🔄 Label Processing
very_positive → positive
very_negative → negative

After normalization, we use:

negative
neutral
positive
⚙️ Project Workflow
1. Load Dataset
from datasets import load_dataset

ds = load_dataset("JosefGoldstein/aimlessinnovations_customer_sentiment_v2")
2. Label Normalization
def normalize_labels(example):
    if example["label_text"] == "very_positive":
        example["label_text"] = "positive"
    elif example["label_text"] == "very_negative":
        example["label_text"] = "negative"
    return example
3. Label Mapping
id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

label2id = {v: k for k, v in id2label.items()}
4. Tokenization
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
5. Model Setup
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)
6. Training
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./bert-sentiment-model",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none"
)
7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)
8. Training
trainer.train()
📈 Evaluation
metrics = trainer.evaluate(test_dataset)
print(metrics)
🔮 Inference
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./bert-sentiment-model",
    tokenizer="./bert-sentiment-model"
)

print(classifier("I really loved this product!"))
💾 Save & Load Model
trainer.save_model("./bert-sentiment-model")
tokenizer.save_pretrained("./bert-sentiment-model")
🚀 Results

After training, the model can classify customer reviews into:

Negative 😡
Neutral 😐
Positive 😊
🧠 Technologies Used
Python 🐍
HuggingFace Transformers 🤗
Datasets Library 📊
PyTorch 🔥
BERT Base Uncased
📌 Notes
Make sure to clean labels before training
Remove duplicated or noisy labels
Ensure num_labels matches your final mapping
Always verify dataset splits before training
📜 License

This project is for educational and research purposes.
