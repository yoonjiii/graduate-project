import os
import json
from urllib.parse import urlparse
import re
from PIL import Image, ImageFont
from io import BytesIO
import requests
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
)

font_path = "fonts/NotoSansKR-Regular.ttf"

def extract_crop_num(url):
    match = re.search(r"/crop(\d+)/", url)
    return int(match.group(1)) if match else float('inf')

def extract_filename(url):
    match = re.search(r'/([^/]+\.(jpg|jpeg|png))', urlparse(url).path, re.IGNORECASE)
    return match.group(1) if match else None

# .gif 파일은 제외하기
# 각 이미지별로, crop#가 순서대로 존재하는지(정렬), 빠진 건 없는지(연속적인지), 혹은 중복되진 않는지 확인
def validate_image_sequence(product):
    print(f"Product Name = {product['product_name']}")
    images = product["images"]

    # gif 파일 제외
    jpg_images = [img for img in images if not urlparse(img).path.lower().endswith(".gif") or not urlparse(img).path.lower().endswith(".webp")]
    product["images"] = jpg_images
    print(f"{len(images)-len(jpg_images)} gif images deleted.")

    file_crop_map = {}
    for img in jpg_images:
        crop_num = extract_crop_num(img)
        filename = extract_filename(img)

        if crop_num != float('inf') and filename:
            file_crop_map.setdefault(filename, []).append(crop_num)
    #print(file_crop_map)

    print(f"-----------<image url checking>-----------")
    accepted = True
    for filename, crop_list in file_crop_map.items():
        crop_list.sort()
        crop_set = set(crop_list)

        max_crop = crop_list[-1]
        expected = set(range(0, max_crop + 1))

        missing = expected - crop_set
        duplicates = [num for num in crop_list if crop_list.count(num) > 1]

        print(f"{filename}, {crop_list}")
        
        if missing or duplicates:
            accepted = False
        if missing:
            print(f"- 누락된 crop 번호: {sorted(missing)}")
        if duplicates:
            print(f"- 중복된 crop 번호: {sorted(set(duplicates))}")

    return product, accepted

def combine_crop_images(product, dirname):
    images = product["images"]
    image_groups = {}

    # 그룹핑: 파일명별로 crop 이미지 묶기
    for img_url in images:
        filename = extract_filename(img_url)
        if not filename:
            continue
        image_groups.setdefault(filename, []).append(img_url)
    
    os.makedirs(dirname, exist_ok=True)
    combined_info = {}
    
    for filename, urls in image_groups.items():
        sorted_urls = sorted(urls, key=extract_crop_num)
        image_parts = []

        for url in sorted_urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
                image_parts.append(img)
            except Exception as e:
                print(f"이미지 로딩 실패: {url} ({e})")

        # 세로로 연결
        if image_parts:
            total_height = sum(img.height for img in image_parts)
            max_width = max(img.width for img in image_parts)
            combined_img = Image.new("RGB", (max_width, total_height))

            y_offset = 0
            for part in image_parts:
                combined_img.paste(part, (0, y_offset))
                y_offset += part.height

            save_path = os.path.join(dirname, filename)
            combined_img.save(save_path)
            combined_info[filename] = save_path

    return combined_info

def download_images(product, dirname):
    images = product["images"]
    image_groups = {}
    
    for img_url in images:
        filename = extract_filename(img_url)
        if not filename:
            continue
        image_groups.setdefault(filename, []).append(img_url)
    
    os.makedirs(dirname, exist_ok=True)
    
    for filename, urls in image_groups.items():
        sorted_urls = sorted(urls, key=extract_crop_num)

        for idx, url in enumerate(sorted_urls):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")

                crop_num = extract_crop_num(url)
                name, ext = os.path.splitext(filename)
                save_name = f"{name}_crop{crop_num:02d}{ext}"
                save_path = os.path.join(dirname, save_name)

                img.save(save_path)
                print(f"Saved: {save_name}")
            except Exception as e:
                print(f"Failed to download: {url} ({e})")

def extract_text(dirname, min_height = 18, draw_vis = True):
    ocr = PaddleOCR(lang='korean', use_angle_cls=False)
    results = {}
    vis_dir = os.path.join(dirname, "ocr_vis")
    if draw_vis:
        os.makedirs(vis_dir, exist_ok=True)

    for filename in sorted(os.listdir(dirname)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(dirname, filename)
            print(f"Processing: {filename}")

            try:
                ocr_result = ocr.ocr(filepath, cls=False)
                texts = []
                boxes = []
                scores = []
                
                if ocr_result and isinstance(ocr_result[0], list):
                    for box, (text, score) in ocr_result[0]:
                        y_coords = [pt[1] for pt in box]
                        height = max(y_coords) - min(y_coords)

                        if height >= min_height:
                            texts.append(text)
                            boxes.append(box)
                            scores.append(score)
                        else:
                            print(f"Skipped small text: '{text}' (height={height})")
                
                original_filename = filename.split("_")[0]
                if original_filename in results :
                    results[original_filename].extend(texts)
                else:
                    results[original_filename] = texts
                print(f"Extracted {len(texts)} lines from {filename}")

                if draw_vis and boxes:
                    image = Image.open(filepath).convert("RGB")
                    image_with_boxes = draw_ocr(np.array(image), boxes, texts, scores, font_path=font_path)
                    vis_image = Image.fromarray(image_with_boxes)
                    vis_image.save(os.path.join(vis_dir, filename))

            except Exception as e:
                print(f"OCR failed for {filename}: {e}")
                results[filename] = []

    full_description = token_join(results)

    json_filename = "ocr_results_" + dirname + ".json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(full_description, f, ensure_ascii=False, indent=4)
    return full_description

# 이미지별로 공백을 두고 텍스트 join.
def token_join(product_detail):
    full_description = {}
    for image, tokens in product_detail.items():
        if len(tokens) > 0 :
            full_description[image] = " ".join(tokens)
    return full_description

def gpt_summarize(full_description, product_N):
    text = ""
    for k, v in full_description.items():
        text += v + " "
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {
                "role": "system",
                "content": "당신은 화장품 상세 정보 이미지 배너에서 OCR로 추출된 정보를 기반으로 텍스트 설명을 제작하는 전문가입니다."
            },
            {
                "role": "user",
                "content": f"""
                    다음은 화장품 상세 설명입니다. 제품의 주요 특징을 최대 3문장으로 요약해주세요.
                    {text}
                    """
            }
        ],
        temperature=0.5
    )

    reply = response.choices[0].message.content
    print(reply)
    return reply


def gpt_highlighted_subjects(full_description, product_N):
    summary = gpt_summarize(full_description, product_N)

    text = ""
    for k, v in full_description.items():
        text += v + " "
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {
                "role": "system",
                "content": "당신은 화장품 상세 정보 이미지 배너에서 OCR로 추출된 정보를 기반으로 텍스트 설명을 제작하는 전문가입니다."
            },
            {
                "role": "user",
                "content": f"""
                    다음은 화장품 제품 설명입니다. 제조사가 강조하고 있는 주요 기능이나 효과를 주제 단위로 정리해주세요.  
                    각 주제는 아래의 JSON 구조를 따라 출력해주세요:

                    출력 형식 예시:
                    {{
                    "features": [
                        {{
                        "keyword": "string",
                        "more_keywords": ["string", "string", ...]
                        "description": "string"
                        }},
                        ...
                    ]
                    }}

                    조건:
                    - "keyword"는 한 단어 또는 짧은 문구로 간결하게
                    - "more_keywords": 겹치는 단어없이, "keyword"의 반의어 하나와, 유사한 의미로 사용될 수 있는 단어 여러 개(1~4개)를 포함해 주세요.
                        예: "수분광택" → ["건조함", "촉촉한", "보습", "글로우"]
                    - "description"은 해당 키워드가 제품에서 의미하는 기능이나 효과를 구체적으로 설명
                    - 4~7개의 주제를 제안해주세요

                    제품 설명:
                    {text}
                    """
            }
        ],
        temperature=0.5
    )

    reply = response.choices[0].message.content
    # 코드 블록 마크다운 제거
    if reply.startswith("```json"):
        reply = reply.lstrip("```json").strip()
    if reply.endswith("```"):
        reply = reply.rstrip("```").strip()
    
    #print(reply)

    try:
        json_result = json.loads(reply)
        json_result["summary"] = summary

        json_filename = "highlighted_subjects_"+product_N+".json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=4)
        print(f"응답이 JSON 파일로 저장되었습니다.")

        keywords = [item['keyword'] for item in json_result.get('features', [])]
        return keywords
    except json.JSONDecodeError as e:
        txt_filename = "highlighted_subjects_"+product_N+".txt"
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(reply)
        print(f"GPT의 요약 응답이 JSON 형식에 맞지 않아, 리뷰 분석을 수행할 수 없습니다.")
        return None

def main():
    # full_dataset은 product의 list
    # product 하나는 dictionary
    # key: product_name, price, images, reviews.

    product_N = "product_0"
    filename = product_N + ".json"

    with open("data/"+filename, "r") as f:
        product = json.load(f)

    accepted = True
    # Data validation
    product, accepted = validate_image_sequence(product)
    if accepted:
        # Data preprocissing
        # combine_crop_images(product, product_N)
        download_images(product, product_N)
        full_description = extract_text(product_N)

        # with open(f"ocr_results_{product_N}.json", "r") as f:
        #     full_description = json.load(f)

        keywords = gpt_highlighted_subjects(full_description, product_N)
        if keywords:
            print(keywords)

    else:
        print("데이터 먼저 정리해주세요.")
        
if __name__ == "__main__":
    main()


