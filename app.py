import fasttext
import gradio as gr
import jieba
from janome.tokenizer import Tokenizer
from pythainlp.tokenize import word_tokenize as thai_tokenize

# Load FastText model
model = fasttext.load_model("models/lang_detect.ftz")

# Setup tokenizers
jpn_tokenizer = Tokenizer()

# Auto-tokenizer for supported langs
def maybe_tokenize(text):
    # Heuristic checks
    contains_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
    contains_hiragana_katakana = any('\u3040' <= c <= '\u30ff' for c in text)
    contains_thai = any('\u0E00' <= c <= '\u0E7F' for c in text)

    # Apply tokenizer by script type
    if contains_cjk:
        return " ".join(jieba.lcut(text))
    elif contains_hiragana_katakana:
        return " ".join([t.surface for t in jpn_tokenizer.tokenize(text)])
    elif contains_thai:
        return " ".join(thai_tokenize(text))
    return text

# Prediction function
def detect_language(text):
    if not text.strip():
        return "Please enter some text."

    processed = maybe_tokenize(text)
    label, confidence = model.predict(processed)
    lang_code = label[0].replace("__label__", "")
    return f"Language: {lang_code}\nConfidence: {confidence[0]*100:.2f}%"

# Gradio Interface
iface = gr.Interface(
    fn=detect_language,
    inputs=gr.Textbox(lines=5, placeholder="Enter text here...", label="Input Text"),
    outputs="text",
    title="Language Detection with FastText",
    description="Enter any sentence, and the model will detect the language using a FastText classifier."
)

if __name__ == "__main__":
    iface.launch(share=True)
