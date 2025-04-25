import streamlit as st
import json
from review_utils import gpt_review_filtering_batched, find_keywords_in_review_with_openai, analyze_sentiment_by_keyword
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

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

def render_summary(summary_text):
    st.markdown("""
        <div style="
            background-color: #fef9f4;
            border-left: 6px solid #F70971;
            padding: 1rem 1.7rem;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        ">
            <h4 style="color: #F70971; margin-top: 0;">📌 제품 요약</h4>
            <p style="font-size: 1.05rem; line-height: 1.6; color: #333;">
                {summary}
            </p>
        </div>
    """.format(summary=summary_text), unsafe_allow_html=True)

st.set_page_config(page_title="화장품 리뷰 비교 분석", layout="wide")

st.title("💄 화장품 제품 설명 vs 사용자 리뷰 비교 분석")
st.markdown("**제조사 설명이 실제 사용자 리뷰에서 어떻게 드러나는지를 키워드 중심으로 분석합니다.**")

selected_product = None

col1, col2, col3, col4 = st.columns(4)

with open(f"data/full_dataset.json", "r", encoding="utf-8") as f:
    products = json.load(f)

if "selected_product" not in st.session_state:
    st.session_state.selected_product = None

with col1:
    st.image(products["product_0"]["image"], width = 300)
    if st.button(f"🧴 {products['product_0']['product_name']}", use_container_width=True):
        st.session_state.selected_product = "product_0"
    st.markdown(f"**{products['product_0']['price']}**")

with col2:
    st.image(products["product_1"]["image"], width = 300)
    if st.button(f"🧴 {products['product_1']['product_name']}", use_container_width=True):
        st.session_state.selected_product = "product_1"
    st.markdown(f"**{products['product_1']['price']}**")

selected_product = st.session_state.selected_product

if selected_product:
    dataset = selected_product
    st.divider()
    st.subheader(f"📦 {products[selected_product]['product_name']} 제품 요약")

    with open(f"data/{dataset}.json", "r", encoding="utf-8") as f:
        product = json.load(f)
        reviews = [r["content"] for r in product["reviews"]]
        st.success(f"{len(reviews)}개의 리뷰가 로드되었습니다.")

    # 키워드 파일
    json_filename = "highlighted_subjects_"+dataset+".json"
    with open(json_filename, "r", encoding="utf-8") as f:
        keyword_data = json.load(f)
    summary = keyword_data['summary']
    keyword_groups = [[item["keyword"]] + item["more_keywords"] for item in keyword_data["features"]]

    if "start_analysis" not in st.session_state:
        st.session_state.start_analysis = False

    if st.button("🔍 리뷰 분석 시작"):
        st.session_state.start_analysis = True
    
    if st.session_state.start_analysis:
        with st.spinner("GPT 전처리 중...(약 10분 소요)"):
            #filtering_output = gpt_review_filtering_batched(reviews, keyword_groups, dataset)

            with open(f"review_filtering_{dataset}.json", "r") as f:
                filtering_output = json.load(f)

            filtered_reviews = [f["filtered"] for f in filtering_output if "filtered" in f]
            original_reviews = [o["original"] for o in filtering_output if "original" in o]

        with st.spinner("임베딩 및 키워드 매칭 중..."):
            keyword_to_reviews = find_keywords_in_review_with_openai(filtered_reviews, keyword_groups)

        with st.spinner("키워드 별 리뷰 분석/요약 중..."):
            keyword_stats = analyze_sentiment_by_keyword(keyword_to_reviews)
            i = 0
            render_summary(summary)
            for kw, sentiments in keyword_stats.items():
                render_analysis_section(kw, keyword_data['features'][i]['description'], sentiments, original_reviews)
                i += 1

        st.success("완료!")