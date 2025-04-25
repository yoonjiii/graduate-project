import streamlit as st
import json
from review_utils import gpt_review_filtering_batched, find_keywords_in_review_with_openai, analyze_sentiment_by_keyword
import time
import pandas as pd
import matplotlib.pyplot as plt

def render_review_with_tooltip(sentence, full_review, score, idx):
    tooltip_html = f"""
    <div style="position: relative; display: inline-block;margin-bottom: 0.6rem;">
        <span style="font-weight: 600; font-size: 1.2rem; color: #333;" title="{full_review}">
            🔹 ({score}) [#{idx}] {sentence}
        </span>
    </div>
    """
    st.markdown(tooltip_html, unsafe_allow_html=True)

def render_review_card(label, reviews, full_reviews, label_color, emoji, ko_label):
    st.markdown(f"""
        <div>
            <h5 style='color:{label_color}; font-size: 1.1rem;'>{emoji} {ko_label} ({len(reviews)}개)</h5>
        </div>
    """, unsafe_allow_html=True)

    # 3개 미리보기 출력
    for sentence, score, idx in reviews[:3]:
        full = full_reviews[idx]
        st.markdown(f"""
            <div style="
                background-color: #f9f9f9;
                border-left: 4px solid {label_color};
                padding: 0.8rem 1rem;
                margin-bottom: 0.6rem;
                border-radius: 8px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                font-size: 1rem;
                ">
                <span title="{full}">
                    <strong>{emoji} ({score}) [#{idx}]</strong> {sentence}
                </span>
            </div>
        """, unsafe_allow_html=True)

    # 전체 리뷰 보기 (expander)
    if len(reviews) > 3:
        with st.expander("전체 리뷰 보기"):
            for sentence, score, idx in reviews[3:]:
                full = full_reviews[idx]
                st.markdown(f"""
                    <div style="
                        background-color: #f9f9f9;
                        border-left: 4px solid {label_color};
                        padding: 0.8rem 1rem;
                        margin-bottom: 0.6rem;
                        border-radius: 8px;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                        font-size: 0.95rem;
                        ">
                        <span title="{full}">
                            <strong>{emoji} ({score}) [#{idx}]</strong> {sentence}
                        </span>
                    </div>
                """, unsafe_allow_html=True)

def render_analysis_section(keyword, description, sentiments, reviews):
    st.subheader(f"💎 키워드: {keyword}")
    st.markdown(f"<p style='margin-top: -0.5rem; margin-bottom: 1rem; color: #555;'>{description}</p>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])

    labels = ["positive", "neutral", "negative"]
    ko_labels = ["좋아요", "그냥 그래요", "별로예요"]
    emojis = ["☺️", "😐", "🙁"]

    sentiment_colors = {
        "positive": "#2ca02c",  # green
        "neutral": "#1f77b4",   # blue
        "negative": "#d62728",  # red
    }

    with left:
        for idx, label in enumerate(labels):
            render_review_card(label, sentiments[label], reviews, sentiment_colors[label], emojis[idx], ko_labels[idx])

    with right:
        counts = [len(sentiments[label]) for label in labels]
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.pie(counts, labels=None, autopct='%1.1f%%', startangle=90, colors=[sentiment_colors[l] for l in labels])
        ax.axis('equal')
        st.pyplot(fig)
    st.markdown("---")

st.set_page_config(page_title="화장품 리뷰 비교 분석", layout="wide")

st.title("💄 화장품 제품 설명 vs 사용자 리뷰 비교 분석")
st.markdown("**제조사 설명이 실제 사용자 리뷰에서 어떻게 드러나는지를 키워드 중심으로 분석합니다.**")

# 리뷰 업로드
uploaded_file = st.file_uploader("📁 리뷰 파일 업로드 (JSON)", type="json")

# 키워드 파일
with open("highlighted_subjects.json", "r", encoding="utf-8") as f:
    keyword_data = json.load(f)

# st.markdown("### 🌟 제품 주요 특성")
# for item in keyword_data["features"]:
#     with st.container():
#         st.markdown(f"#### 🔹 {item['keyword']}")
#         st.markdown(f"- **설명:** {item['description']}")
#         st.markdown(f"- **연관 키워드:** {', '.join(item['more_keywords'][1:])}")
#         st.markdown("---")

# st.markdown("### 💎 주요 특성 카드 뷰")
# cols = st.columns(2)
# for i, item in enumerate(keyword_data["features"]):
#     with cols[i % 2]:
#         st.markdown(f"""
#         #### 🔹 {item['keyword']}
#         - **설명:** {item['description']}

#         - **연관 키워드:** {", ".join(item['more_keywords'][1:])}
#         """)


if uploaded_file:
    product = json.load(uploaded_file)
    reviews = [r["content"] for r in product["reviews"]]
    st.success(f"{len(reviews)}개의 리뷰가 로드되었습니다.")
    keyword_groups = [[item["keyword"]] + item["more_keywords"] for item in keyword_data["features"]]

    if st.button("리뷰 분석 시작"):
        with st.spinner("GPT 전처리 중...(약 10분 소요)"):
            #filtering_output = gpt_review_filtering_batched(reviews, keyword_groups)

            time.sleep(5)
            with open("review_filtering.json", "r") as f:
                filtering_output = json.load(f)
            filtered_reviews = [f["filtered"] for f in filtering_output if "filtered" in f]

        with st.spinner("임베딩 및 키워드 매칭 중..."):
            keyword_to_reviews = find_keywords_in_review_with_openai(filtered_reviews, keyword_groups)
            # 출력
            # for keyword, matched_reviews in keyword_to_reviews.items():
            #     st.subheader(f"🔍 키워드: {keyword} (총 {len(matched_reviews)} 문장)")
            #     for i, (sentence, score, idx) in enumerate(sorted(matched_reviews, key=lambda x: x[1], reverse=True)):
            #         st.markdown(f"{i+1}. **({score})** 리뷰 #{idx} - {sentence}")

        with st.spinner("키워드 별 리뷰 분석/요약 중..."):
            keyword_stats = analyze_sentiment_by_keyword(keyword_to_reviews)
            i = 0
            for kw, sentiments in keyword_stats.items():
                render_analysis_section(kw, keyword_data['features'][i]['description'], sentiments, reviews)
                i+=1

        st.success("완료!")

        
