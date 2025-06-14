import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Define the summarization function
def summarize(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Create Gradio interface
interface = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, placeholder="Enter text to summarize here..."),
    outputs="text",
    title="Text Summarizer",
    description="Summarize long documents using Facebook BART model"
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
