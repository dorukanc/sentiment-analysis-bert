import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the trained model and tokenizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("my_awesome_model/checkpoint-150").to(device)
tokenizer = AutoTokenizer.from_pretrained("my_awesome_model/checkpoint-150")

# Function to make predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

# Gradio Interface
iface = gr.Interface(
    fn=predict, 
    inputs=gr.Textbox(label="Enter a comment"), 
    outputs="text",
    live=False,  # Set live=False so that the user has to submit input (no live updates)
    title="Sentiment Classifier",
    description="Type a comment and the model will classify whether the sentiment is Positive or Negative.",
    theme="compact",  # Optional: you can choose a theme for a cleaner UI
)

# Launch the interface
iface.launch()
