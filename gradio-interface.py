import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the trained model and tokenizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("dorukan/distilbert-base-uncased-bert-finetuned-imdb").to(device)
tokenizer = AutoTokenizer.from_pretrained("dorukan/distilbert-base-uncased-bert-finetuned-imdb")

# Function to make predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

# Create custom CSS for the footer
css = """
footer {
    text-align: center;
    padding: 20px;
    border-top: 1px solid #ddd;
    margin-top: 20px;
}
footer a {
    color: #007bff;
    text-decoration: none;
    margin-left: 5px;
}
footer a:hover {
    text-decoration: underline;
}
"""

# Create custom footer HTML
footer_html = """
<footer>
    <p>Made with ❤️ by Dorukan | 
    <a href="https://github.com/dorukanc" target="_blank">GitHub</a>
    </p>
</footer>
"""

# Gradio Interface
iface = gr.Interface(
    fn=predict, 
    inputs=gr.Textbox(label="Enter a comment"), 
    outputs="text",
    live=False,
    title="Sentiment Classifier w/ DistilBERT",
    description="Type a comment and the model will classify whether the sentiment is Positive or Negative.",
    theme="compact",
    css=css,
    article=footer_html  # Add the footer using the article parameter
)

# Launch the interface
iface.launch()