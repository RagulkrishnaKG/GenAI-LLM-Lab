from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

texts = [
    "The product is excellent",
    "I am very happy with the service",
    "The product is terrible",
    "This was a disappointing experience",
]
labels = [1, 1, 0, 0]

dataset = Dataset.from_dict({"text": texts, "label": labels})

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
trainer.save_model("./fine_tuned_model")
print("Fine-tuning completed successfully.")
