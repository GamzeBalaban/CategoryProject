from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import uvicorn
import re
import string
import contractions
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

# NLTK bileşenleri
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

# FastAPI uygulaması
app = FastAPI()

# BERT model yükleme
model_path = "./bert_department_model"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Cihaz tanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ağırlıkları yükle
try:
    model.load_state_dict(torch.load("bert_department_model.pt", map_location=device))
    print("Model ağırlıkları başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")

model.to(device)
model.eval()

# Etiket eşlemesi
label_mapping = {
    0: "Trend",
    1: "Jackets",
    2: "Intimate",
    3: "Bottoms",
    4: "Dresses",
    5: "Tops"
}

# POS etiket dönüşümü
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Temizleme fonksiyonu
def clean_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [t for t in words if t not in stop_words and t not in punctuations and t.isalpha()]
    pos_tags = pos_tag(words)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(lemmatized)

# İstek modeli
class ReviewRequest(BaseModel):
    text: str

# Tahmin endpoint'i
@app.post("/predict")
def predict(review: ReviewRequest):
    cleaned_text = clean_text(review.text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return {
        "department": label_mapping.get(prediction, "Unknown"),
        "cleaned_text": cleaned_text
    }

# Ana çalıştırma
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
