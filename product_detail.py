import os
import json
from urllib.parse import urlparse
import re
from PIL import Image
from io import BytesIO
import requests

def extract_crop_num(url):
    match = re.search(r"/crop(\d+)/", url)
    return int(match.group(1)) if match else float('inf')

def extract_filename(url):
    match = re.search(r'/([^/]+\.(jpg|jpeg|png))', urlparse(url).path, re.IGNORECASE)
    return match.group(1) if match else None

# .gif 파일은 제외하기
# 각 이미지별로, crop#가 순서대로 존재하는지(정렬), 빠진 건 없는지(연속적인지), 혹은 중복되진 않는지 확인
def validate_image_sequence(product):
    print(f"Product Name = {product["product_name"]}")
    images = product["images"]

    # gif 파일 제외
    jpg_images = [img for img in images if not urlparse(img).path.lower().endswith(".gif")]
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


def combine_crop_images(product):
    images = product["images"]
    image_groups = {}

    # 그룹핑: 파일명별로 crop 이미지 묶기
    for img_url in images:
        filename = extract_filename(img_url)
        if not filename:
            continue
        image_groups.setdefault(filename, []).append(img_url)
    
    os.makedirs("combined_images", exist_ok=True)
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

            save_path = os.path.join("combined_images", filename)
            combined_img.save(save_path)
            combined_info[filename] = save_path

    return combined_info


def main():
    # full_dataset은 product의 list
    # product 하나는 dictionary
    # key: product_name, price, images, reviews.
    with open("product_0.json", "r") as f:
        product = json.load(f)

    # Data validation
    product, accepted = validate_image_sequence(product)
    if accepted:
        # Data preprocissing
        combine_crop_images(product)
    else:
        print("데이터 먼저 정리해주세요.")
        
if __name__ == "__main__":
    main()