import gradio as gr
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    if not text.strip():
        return "Please enter text to summarize."
    result = summarizer(text, max_length=80, min_length=20, do_sample=False)
    return result[0]["summary_text"]

app = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, label="Input Text"),
    outputs=gr.Textbox(label="Generated Summary"),
    title="Generative AI Text Summarizer",
    description="Enter a paragraph and generate a concise summary using a pre-trained transformer model.",
)

if __name__ == "__main__":
    app.launch()
