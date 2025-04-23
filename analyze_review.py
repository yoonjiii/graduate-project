import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from kiwipiepy import Kiwi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

load_dotenv()
kiwi = Kiwi()

model_name = "nlp04/korean_sentiment_analysis_kcelectra"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
)

POSITIVE_LABELS = {0, 1, 2, 3, 4}
NEGATIVE_LABELS = {7, 8, 9, 10}
NEUTRAL_LABELS = {5, 6}

def remove_symbols(text):
    text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", "", text)  # ㅋㅋ, ㅠㅠ 등 제거
    text = re.sub(r"[^\w\s가-힣.,!?]", "", text)  # 이모티콘, 기타 기호 제거
    return text.strip()

def insert_period(text): # 종결어미 뒤에 문장부호가 없는 경우, 마침표 추가.
    return re.sub(r"(니다|어요|아요|해요|네요|군요|구요|고요|여요|려요)(?=\s[^\.\!\?])", r"\1.", text)

def split_sentences(text):
    return [s.text.strip() for s in kiwi.split_into_sents(text)]

def map_to_sentiment(label_id):
    if label_id in POSITIVE_LABELS:
        return "positive"
    elif label_id in NEGATIVE_LABELS:
        return "negative"
    else:
        return "neutral"

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()
        return map_to_sentiment(label_id)

def find_keywords_in_review(reviews, keywords):
    sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    keyword_stats = {kw: {"positive": [], "negative": [], "neutral": []} for kw in keywords}

    keywords_vec = [(kw, sbert_model.encode(kw, normalize_embeddings=True)) for kw in keywords]

    for review in reviews:
        sentence_vec = sbert_model.encode(review, normalize_embeddings=True)

        for keyword, vec in keywords_vec:
            sim_score = util.cos_sim(vec, sentence_vec)
            if sim_score >= 0.5:
                sentiment = classify_sentiment(review)
                keyword_stats[keyword][sentiment].append(review)
    return keyword_stats



def gpt_analyze_review(reviews, keywords):
    preprocessed_reviews = []
    for review_obj in reviews:
        content = review_obj["content"]
        cleaned = remove_symbols(content)
        sentence = insert_period(cleaned)
        sentences = split_sentences(sentence)
        preprocessed_reviews.extend(sentences)

    keyword_stats = find_keywords_in_review(preprocessed_reviews, keywords)

    # 여기서 keyword_stats에 대해서 추가 분석 예정.
    for keyword, sentiments in keyword_stats.items():
        total = sum(len(sentiments[sent]) for sent in sentiments)
        print(f"\n🔍 키워드: {keyword} (총 {total} 문장)")
        for label, reviews in sentiments.items():
            print(f"  - {label.capitalize()}: {len(reviews)}개")
            for example in reviews[:1]:
                print(f"    예시: {example}")


def main():
    product_N = "product_0"
    filename = product_N + ".json"

    with open(filename, "r") as f:
        product = json.load(f)   #product의 상세이미지 url, 100개의 리뷰가 들어있음.
    
    keywords = ['수분광택', '초밀착커버', '스킨케어 효과', '다양한 컬러', '가벼운 텍스처', '지속력', '편리한 사용']
    if keywords:
        gpt_analyze_review(product['reviews'], keywords)
        
if __name__ == "__main__":
    main()