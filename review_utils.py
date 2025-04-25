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
from collections import defaultdict
import numpy as np

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

def string_match(keyword, sentence):
    words = keyword.split()
    for word in words:
        for i in range(len(word) - 1):
            sub = word[i:i+2]
            if sub in sentence:
                return True
    return False

def find_keywords_in_review(reviews, keyword_groups):
    sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    keyword_to_reviews = defaultdict(set)
    keyword_vecs = {}
    keyword_dict = {}
    for group in keyword_groups:
        main_kw = group[0]
        keyword_vecs[main_kw] = [sbert_model.encode(kw, normalize_embeddings=True) for kw in group]
        keyword_dict[main_kw] = group[1:]

    for idx, review in enumerate(reviews):
        cleaned = remove_symbols(review)
        sentence = insert_period(cleaned)
        sentences = split_sentences(sentence)
        
        for sentence in sentences:
            sentence_vec = sbert_model.encode(sentence, normalize_embeddings=True)

            for main_kw, vec_list in keyword_vecs.items():
                scores = [util.cos_sim(vec, sentence_vec).item() for vec in vec_list]
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                
                if (max_score >= 0.5 and avg_score >= 0.35)\
                    or any(string_match(alt_kw, sentence) for alt_kw in keyword_dict[main_kw]):
                    keyword_to_reviews[main_kw].add((sentence, round(avg_score, 2), idx))
    
    for kw in keyword_to_reviews:
        keyword_to_reviews[kw] = sorted(keyword_to_reviews[kw], key=lambda x: x[1], reverse=True)
    print(keyword_to_reviews)
    return keyword_to_reviews

def get_embedding(text: str, model="text-embedding-3-small") -> list:
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def batch_get_embeddings(text_list: list[str], model="text-embedding-3-small", batch_size = 50) -> list:
    all_embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def find_keywords_in_review_with_openai(reviews, keyword_groups):
    keyword_to_reviews = defaultdict(set)
    keyword_vecs = {}
    keyword_dict = {}

    for group in keyword_groups:
        main_kw = group[0]
        keyword_vecs[main_kw] = [get_embedding(kw) for kw in group]
        keyword_dict[main_kw] = group[:]

    sentence_info = [] #(sentence, idx)
    for idx, review in enumerate(reviews):
        cleaned = remove_symbols(review)
        #sentence = insert_period(cleaned)
        sentences = split_sentences(cleaned)
        for s in sentences:
            if len(s) < 4 and len(s.split()) <= 1:
                continue
            sentence_info.append((s, idx))

    all_sentences = [s for s, _ in sentence_info]
    all_embeddings = batch_get_embeddings(all_sentences)

    for (sentence, idx), sentence_vec in zip(sentence_info, all_embeddings):

        scores = [] #각 키워드의 맥스값을 저장했다가, 마지막에 그중 가장 큰 값을 가지는 키워드와 매칭.
        matched = [] #최종 매칭
        for main_kw, vec_list in keyword_vecs.items():
            max_score = max([cosine_similarity(vec, sentence_vec) for vec in vec_list])
            if max_score >= 0.3:
                scores.append((max_score, main_kw))
                if any(string_match(alt_kw, sentence) for alt_kw in keyword_dict[main_kw]): #이 경우, 바로 최종 매칭에 추가.
                    matched.append((max_score, main_kw))
        if scores:
            top_match = max(scores)
            if top_match not in matched:
                matched.append(top_match)
        
        for (max_score, main_kw) in matched:
            keyword_to_reviews[main_kw].add((sentence, round(max_score, 2), idx))
    
    return keyword_to_reviews

def review_sentiment_analysis(keyword_to_reviews, keywords):
    keyword_stats = {kw: {"positive": [], "negative": [], "neutral": []} for kw in keywords}
    for keyword, matched in keyword_to_reviews.items():
        for sentence, idx in matched:
            sentiment = classify_sentiment(sentence)  # 예: "positive"
            keyword_stats[keyword][sentiment].append((sentence, idx))
    return keyword_stats

def gpt_review_filtering(reviews, keyword_groups):
    main_keywords = [group[0] for group in keyword_groups]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {
                "role": "system",
                "content": "당신은 화장품의 온라인 쇼핑몰 사용자 리뷰 전처리 전문가입니다."
            },
            {
                "role": "user",
                "content": f"""
                    아래는 전처리가 필요한 소비자 리뷰 목록이며, 불필요한 내용을 조금만 필터링하려고 합니다. 각 리뷰마다 다음 작업을 수행해 주세요:
                        **주의: {main_keywords}에 있는 키워드와 연관된 내용이 누락되지 않도록 한다.**
                        1. 제품과 무관한 내용(제품 사용 전 기대, 가격/배송/브랜드 관련 코멘트 등)을 제외해 주세요.
                        2. 오탈자 및 문장부호 오류 수정.
                        3. 각 리뷰는 JSON 구조로, 다음과 같은 형식을 따라주세요:
                        [
                          {{
                            "original": "원본 리뷰 내용",
                            "filtered": "필요 문장만 남은 내용"
                          }},
                          ...
                        ]
                    리뷰 목록: {reviews}
                    """
            }
        ],
        temperature=0.5
    )

    reply = response.choices[0].message.content
    #print(reply)

    if reply.startswith("```json"):
        reply = reply.lstrip("```json").rstrip("```").strip()
    elif reply.startswith("```"):
        reply = reply.lstrip("```").rstrip("```").strip()

    reply = re.sub(r'\\U[0-9A-Fa-f]{8}', '', reply)

    try:
        filtered = json.loads(reply)

        with open("review_filtering.json", "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=4)

        return filtered
    except Exception as e:
        print("리뷰필터링 - GPT 응답 파싱 실패:", e)
        with open("review_filtering.txt", "w", encoding="utf-8") as f:
            f.write(reply)
        return reviews  # fallback

def gpt_review_filtering_batched(reviews, keyword_groups, batch_size=20):
    main_keywords = [group[0] for group in keyword_groups]
    filtered = []

    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 화장품의 온라인 쇼핑몰 사용자 리뷰 전처리 전문가입니다."
                },
                {
                    "role": "user",
                    "content": f"""
                        아래는 전처리가 필요한 소비자 리뷰 목록입니다. 각 리뷰마다 다음 작업을 수행해 주세요:
                        **주의: {main_keywords}에 있는 키워드와 연관된 내용이 누락되지 않도록 한다.**
                        1. 제품과 무관한 내용(기대, 가격, 배송 등)을 제외
                        2. 오탈자 및 문장부호 오류 수정
                        3. 지나친 구어체 수정
                        4. 아래 형식으로 출력:
                        ```json
                        [
                          {{
                            "original": "원본 리뷰",
                            "filtered": "필요 문장만 남은 내용"
                          }},
                          ...
                        ]
                        ```
                        리뷰 목록: {batch}
                    """
                }
            ],
            temperature=0.5
        )

        reply = response.choices[0].message.content

        if reply.startswith("```json"):
            reply = reply.lstrip("```json").rstrip("```").strip()
        elif reply.startswith("```"):
            reply = reply.lstrip("```").rstrip("```").strip()

        reply = re.sub(r'\\U[0-9A-Fa-f]{8}', '', reply)

        try:
            batch_filtered = json.loads(reply)
            filtered.extend(batch_filtered)
        except Exception as e:
            print(f"Batch {i // batch_size + 1} 파싱 실패:", e)
            # with open(f"review_filtering_batch_{i}.txt", "w", encoding="utf-8") as f:
            #     f.write(reply)
            # filtered.extend([{"original": r, "filtered": r} for r in batch])  # fallback

    with open("review_filtering.json", "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=4)

    return filtered

def analyze_sentiment_by_keyword(keyword_to_reviews):
    """
    keyword_to_reviews: Dict[str, Set[Tuple[str, float, int]]]
    반환: Dict[str, Dict[str, List[Tuple[str, float, int]]]]
        예: keyword_stats["수분광택"]["positive"] = [(문장, 점수, 인덱스), ...]
    """
    keyword_stats = defaultdict(lambda: {"positive": [], "neutral": [], "negative": []})

    for keyword, review_set in keyword_to_reviews.items():
        for sentence, score, idx in review_set:
            sentiment = classify_sentiment(sentence)
            keyword_stats[keyword][sentiment].append((sentence, score, idx))

    return keyword_stats

def main():
    # 리뷰 불러오기
    product_N = "product_0"
    filename = product_N + ".json"

    with open(filename, "r") as f:
        product = json.load(f)   #product의 상세이미지 url, 100개의 리뷰가 들어있음.

    reviews = []
    for review_obj in product['reviews']:
        reviews.append(review_obj["content"])
    
    # 키워드 불러오기
    with open("highlighted_subjects.json", "r") as f:
        keyword_data = json.load(f)

    keyword_groups = [[item["keyword"]] + item["keyword_synonyms"] for item in keyword_data["features"]]

    filtering_output = gpt_review_filtering_batched(reviews, keyword_groups)

    # with open("review_filtering.json", "r") as f:
    #     filtering_output = json.load(f)

    filtered_reviews = [item["filtered"] for item in filtering_output if "filtered" in item]
    keyword_to_reviews = find_keywords_in_review_with_openai(filtered_reviews, keyword_groups)

    print(keyword_to_reviews)
    print()

    for keyword, review_set in keyword_to_reviews.items():
        sorted_reviews = sorted(review_set, key=lambda x: x[1], reverse=True)
        print(f"\n🔍 키워드: {keyword} (총 {len(sorted_reviews)} 문장)")
        for i, (sentence, score, idx) in enumerate(sorted_reviews):
            print(f"  {i}. ({score}) [리뷰 #{idx}] {sentence}")


        
if __name__ == "__main__":
    main()