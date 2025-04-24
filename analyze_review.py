import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
import openai
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
        sentence = insert_period(cleaned)
        sentences = split_sentences(sentence)
        for s in sentences:
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
                        3. 지나친 구어체 개선
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

def analyze_review(reviews, keyword_groups):

    ex_reviews = [
        "피부타입 복합성 색상 > 촉촉함 > 커버력 순으로 고려하는데용 색상은 밝은 21호 옐로톤에 딱 좋았구요! 커버력 좋은 제형이라서 부분적인 잡티기 잘 가려지네요 ! 바르면 촉촉하다 이런 느낌은 아니라서 촉촉하게 바르고 싶다하시면 스킨케어를 촉촉하게 하면 될 것 같아요 바르고 수정하지 않았을 때 잔주름이랑 모공에 끼임이 조금 있는 편이었어요 여름에 쓰기 진짜 좋을 것 같아요 ! ㅎㅎ",
        "매장에서 테스트 했을 때 좋길래 이번 라이브 방송 때 구매했습니다! 음,, 양 조절 필수입니다,,,,, 바르기 어려워요ㅠㅠ 얇게 발리는 만큼 커버력은 기대하지 않는 게 좋을 것 같습니다! 하지만~! 자연스러운 커버를 하기엔 넘 좋다는 거! 첨 발랐을 땐 촉촉한 듯 하나 금방 건조해져용 ㅠㅠ 지성이 겨울에 쓰기 딱 좋은 정도!! 건성이신 분들은 건조하다고 느껴질 거 같아요! 그리고 파우더 처리 제대로 안 하면 바로 밀립니다ㅠ",
        "리뉴얼 전 제품을 너무 잘 써서 요 제품 너무 궁금해서 구매해봤습니다..! 아직 한번 사용한거라 막 이렇다 저렇다 말하긴 어렵지만, 기존 2 란제리 컬러보다 붉은기가 빠지고 화사해졌어요! 저는 부분부분 붉은기가 있는 피부톤이라 기존 제품 사용하면서 붉은기 커버가 안되는게 살짝 아쉬웠는데, 상아빛 뉴트럴 아이보리컬러라 붉은기 커버 됩니댜...! 커버력이나 지속력도 나쁘지않았고, 다만 크게 촉촉한 쿠션은 아니예요...! ( 하지만 클리오 쿠션 라인중 촉촉한 편 ! ) 겨울철보다는 봄 ~ 여름에 사용하기 좋을듯합니다!  (+) 추가 흠..계속 쓰다보니 느낀건데...파운웨어 디 오리지널이 더 촉촉한 느낌이예요....!",
        "💙제형 다른 쿠션들과 비교했을 때 훨씬 물 같은 제형이고 시원 촉촉하게 발립니다!  💜색상 디오리지널 쿠션보다 자연스러운 색상이라 좋았어요 제가 샀던 색이 란제리 21c 색이라 그런지는 모르겠는데 저번에 디 오리지널 쿠션을 샀을 땐 주황끼가 많이 돌아서 좀 별로였거든요... 근데 메쉬글로우 쿠션은 자연스러운 색상이라 좋았습니당 (색상비교는 사진 참고해주세요)  🩷장점 - 메쉬망이라 골고루 퍼프에 묻어나오고 양조절이 쉬움 - 퍼프가 바르기 쉽게 만들어진 모양이라 좋음 - 글로우 치고는 커버력이 꽤 있는 편! (잡티커버보다는 홍조커버)   ❤️단점 - 요철부각 약간 (피부타입에 따라 다를 수 있음) - 개인적으로 케이스 디자인이 별루.. - 잘 벗겨지고 묻어남(파우더나 픽서로 고정필수)   ✨️추천 - 글로우한 쿠션을 좋아하지만 과한 광은 싫어하시는 분 - 편하게 바르기 좋은 데일리 글로우 쿠션 찾으시는 분 - 피부 트러블이 별로 없으신 분  🖤사진 1. 타쿠션과 색상 비교 2. 첫번째 사진에서 클리오쿠션만 약간 왜곡되어 찍힌 것 같아서 올려봐요! 순서는 앞 사진과 동일합니다 3. 구성품 치크 발색샷",
        "라방가격 25,500원/ 매장오특인지 포스기 앞에 24,900원이라고 되었더라고요.(매장별로 다른지는 모르겠지만) 라방하자마자 쿠폰써서 결제하고 픽업하러갔는데 가격 차이가 있어서 고작 몇백원 차이지만 놀아난느낌 계속 이용하는 저도 뭐 호구로써 할말은 없는데 그냥 먼느낌인지 아시죠..ㅋㅋ  메쉬쿠션 쓰고 싶었는데 마침 리뉴얼됬다고 해서 구매해봤어요. 클리오랑 너무 안 맞아서 매번 실패하고 걸렀는데 이번에는 기대하고 있어요.  + 추가 리뷰 (개인차가 있으니 참고만 해주세요.) 피부타입: 속건조 심한 건성. 염증성트러블피부.각질. 항상 기초왕빵빵하게 하는편  생각보다 피부가 촉촉한 느낌의 촉촉글로우는 아니었고 마무리만 글로이하게 되는 제품같은데, 제 피부에서는 세미글로우정도로 표현됐어요. 글로우치고 커버력이 있어요. 턱턱 잘 쌓이는 제형이고 수정하기도 좋은데 밀착력이 엄청 좋다는 느낌은 못 받았어요. 입가제외 건조해지거나 당긴다는 느낌은 없어서 크게 불편하거나 답답하지 않았어요. 입가,트러블주위 각질 부각이 있었고, 모공/요철부각은 없었어요. 머리카락 붙음 적었어요. 지성이나 건성인데 스킨케어 빵빵하게 안한다하면 비추 그외 피부타입은 두루두루 쓰기 좋을 것 같아요. 👍🏻저는 이번 클리오 쿠션 나름 성공적이었어요.",
        "다른 브랜드 파데 쓰는데 너무 건조해서 뜨는 바람에 사게된 쿠션이에요 건성한테 좋대서 사봤어용",
        "신기하게 생겨서 샀는데 커버도 잘되고 딱 적당하게 촉촉해서 화장이 뜨지도않아요!!"
    ]

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

    # # 여기서 keyword_stats에 대해서 추가 분석 예정.
    # for keyword, sentiments in keyword_stats.items():
    #     total = sum(len(sentiments[sent]) for sent in sentiments)
    #     print(f"\n🔍 키워드: {keyword} (총 {total} 문장)")
    #     for label, reviews in sentiments.items():
    #         print(f"  - {label.capitalize()}: {len(reviews)}개")
    #         print(f"    예시: ")
    #         for example, score in reviews:
    #             print(f"    ({score:.2f}){example}")


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
        subjects = json.load(f)

    keyword_groups = []
    for feature in subjects["features"]:
        keyword_groups.append([feature["keyword"]] + feature["keyword_synonyms"])

    analyze_review(reviews, keyword_groups)
        
if __name__ == "__main__":
    main()