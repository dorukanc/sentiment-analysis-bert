from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import datasets
import torch
import os


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

imdb = load_dataset("csv", data_files="dataset/IMDB.csv", split="train")
imdb = imdb.train_test_split(test_size=0.2) # 80 % train 20 % test

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"], 
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


'''

# After training, save the model 
model.save_pretrained("awesome_model")
tokenizer.save_pretrained("awesome_tokenizer")

save_path = os.path.join(os.getcwd(), "first.pt")
output = open(save_path, mode="wb")
torch.save({'model_state_dict':model.state_dict()},output)

text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
inputs = tokenizer(text, return_tensors="pt").to(device)
logits = model(**inputs).logits
print("LOGITS: ", logits)
predicted_class_id = logits.argmax().item()
print("RESULT: ", model.config.id2label[predicted_class_id])


state_dict = torch.load(f=save_path, map_location=torch.device(device))
model.load_state_dict(state_dict['model_state_dict'])

inputs = tokenizer(text, return_tensors="pt").to(device)
logits2 = model(**inputs).logits
print("LOGITS2: ", logits2)
predicted_class_id2 = logits2.argmax().item()
print("RESULT2: ", model.config.id2label[predicted_class_id2])

'''

