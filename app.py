import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import re
import json

# Cấu hình Streamlit
st.set_page_config(page_title="Better Translation for Vietnamese", layout="wide")

# Tải mô hình dịch máy
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Tải từ điển đơn giản
@st.cache_resource
def load_simple_dict():
    with open('simple_dict.json', 'r', encoding='utf-8') as f:
        return json.load(f)

simple_dict = load_simple_dict()

# Hàm chuẩn hóa văn bản
# Hàm chuẩn hóa văn bản (cập nhật để giữ dấu phẩy)
def normalize_text_for_evaluation(text):
    # Loại bỏ ký tự không cần thiết, chuẩn hóa khoảng trắng
    text = re.sub(r'[^\w\s,.!?]', '', text)  # Chỉ giữ lại chữ, số, dấu câu
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa
    text = text.strip()

    # Đảm bảo mỗi câu kết thúc bằng dấu chấm (nếu thiếu)
    if not text.endswith(('.', '!', '?')):
        text += '.'

    # Tách câu dựa vào dấu câu, giữ dấu phẩy trong câu
    sentences = re.split(r'(?<=[.!?])\s*', text)  # Dấu câu theo sau là khoảng trắng hoặc kết thúc
    normalized_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # Bỏ qua câu rỗng
            normalized_sentences.append(sentence.capitalize())

    # Ghép các câu lại với dấu cách sau mỗi câu
    return ' '.join(normalized_sentences).strip()


# Hàm dịch văn bản
def translate(text, model, tokenizer):
    if not text.strip():
        return "Please enter some text to translate."

    words = text.strip().split()

    if len(words) == 1:
        word = words[0].lower()
        if word in simple_dict:
            return simple_dict[word]
        else:
            inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            translated = model.generate(**inputs)
            return tokenizer.decode(translated[0], skip_special_tokens=True)

    processed_words = [simple_dict.get(word.lower(), word) for word in words]
    simplified_text = " ".join(processed_words)

    inputs = tokenizer(simplified_text.strip().lower(), return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(
        **inputs,
        num_beams=35,
        no_repeat_ngram_size=2,
        length_penalty=1.0,
        early_stopping=True
    )
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

# Hàm tính F1 Score
def calculate_f1_score(translated_text, reference_text):
    reference_text = normalize_text_for_evaluation(reference_text)
    translated_text = normalize_text_for_evaluation(translated_text)

    reference_tokens = reference_text.split()
    translated_tokens = translated_text.split()
    common_tokens = set(reference_tokens) & set(translated_tokens)

    precision = len(common_tokens) / len(translated_tokens) if translated_tokens else 0
    recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

# Giao diện Streamlit
st.title("Better Translation for Vietnamese")

english_text = st.text_area("Enter text in English:", placeholder="Type your English text here...")
reference_text = st.text_area("Enter reference translation in Vietnamese:", placeholder="Type the reference translation here...")


if st.button("Translate and Evaluate"):
    with st.spinner("Translating..."):
        vietnamese_text = translate(english_text, model, tokenizer)
        
        # Chuẩn hóa văn bản đã dịch
        normalized_vietnamese_text = normalize_text_for_evaluation(vietnamese_text)
        
        # Hiển thị văn bản đã dịch (đã chuẩn hóa)
        st.text_area("Translated text in Vietnamese:", normalized_vietnamese_text, height=200)

        if reference_text.strip():
            f1 = calculate_f1_score(normalized_vietnamese_text, reference_text)
            st.write(f"**F1 Score (Current):** {f1:.2f}")

            if "results" not in st.session_state:
                st.session_state.results = []

            st.session_state.results.append(f1)
            avg_f1 = sum(st.session_state.results) / len(st.session_state.results)
            st.write(f"**Average F1 Score:** {avg_f1:.2f}")
