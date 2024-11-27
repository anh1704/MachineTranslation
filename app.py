import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import re
import json

# Danh sách toàn cục để lưu F1 scores
if "results" not in st.session_state:
    st.session_state.results = []

# Cấu hình giao diện
st.set_page_config(
    page_title="Better Translation for Vietnamese",
    layout='wide'
)

# Tải mô hình dịch máy
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# # Tải từ điển đơn giản
# @st.cache_resource
# def load_simple_dict():
#     with open('simple_dict.json', 'r') as f:
#         simple_dict = json.load(f)
#     return simple_dict

# simple_dict = load_simple_dict()

# Chuẩn hóa văn bản
def normalize_text_for_evaluation(text):
    text = re.sub(r'\b(anh|chị|bạn|em)\b', 'bạn', text, flags=re.IGNORECASE)  # Chuẩn hóa từ xưng hô
    return text.strip().lower()

# Hàm dịch
def translate(text, model, tokenizer):
    if not text.strip(): 
        return "Please enter some text to translate."
    
    # Tiến hành dịch
    inputs = tokenizer(text.strip().lower(), return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(
        **inputs,
        num_beams=10,
        no_repeat_ngram_size=3,
        length_penalty=1.2,
        early_stopping=True
    )
    translation = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translation

# Tính F1 Score
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
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Giao diện Streamlit
st.title("Better Translation for Vietnamese")

english_text = st.text_area("Enter text in English:", placeholder="Type your English text here...")
reference_text = st.text_area("Enter reference translation in Vietnamese:", placeholder="Type the reference translation here...")

if st.button("Translate and Evaluate"):
    with st.spinner("Translating..."):
        vietnamese_text = translate(english_text, model, tokenizer)
        st.text_area("Translated text in Vietnamese:", vietnamese_text, height=200)
        
        if reference_text.strip():
            f1 = calculate_f1_score(vietnamese_text, reference_text)
            st.write(f"**F1 Score (Current):** {f1:.2f}")
            
            # Lưu điểm F1 vào session_state và tính trung bình
            st.session_state.results.append(f1)
            avg_f1 = sum(st.session_state.results) / len(st.session_state.results)
            st.write(f"**Average F1 Score:** {avg_f1:.2f}")