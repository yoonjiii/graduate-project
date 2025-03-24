import json
from urllib.parse import urlparse
import re

# images 전처리
# .gif 파일은 제외하기
# 각 이미지별로, crop#가 순서대로 존재하는지(정렬), 빠진 건 없는지(연속적인지), 혹은 중복되진 않는지 확인
def validate_image_sequence(dataset):
    for product in dataset:
        print(f"Product Name = {product["product_name"]}")
        images = product["images"]

        # gif 파일 제외
        jpg_images = [img for img in images if not urlparse(img).path.lower().endswith(".gif")]
        product["images"] = jpg_images
        print(f"{len(images)-len(jpg_images)} gif images deleted.")

        file_crop_map = {}
        for img in jpg_images:
            # crop 번호 추출
            crop_match = re.search(r"/crop(\d+)/", img)
            # 파일명 추출
            file_match = re.search(r'/([^/]+\.(jpg|jpeg|png))', urlparse(img).path, re.IGNORECASE)

            if crop_match and file_match:
                crop_num = int(crop_match.group(1))
                filename = file_match.group(1)

                file_crop_map.setdefault(filename, []).append(crop_num)
        #print(file_crop_map)

        print(f"-----------<image url checking>-----------")
        for filename, crop_list in file_crop_map.items():
            crop_list.sort()
            crop_set = set(crop_list)

            min_crop = 0
            max_crop = crop_list[-1]
            expected = set(range(min_crop, max_crop + 1))

            missing = expected - crop_set
            duplicates = [num for num in crop_list if crop_list.count(num) > 1]

            print(f"{filename}, {crop_list}")

            if missing:
                print(f"- 누락된 crop 번호: {sorted(missing)}")
            if duplicates:
                print(f"- 중복된 crop 번호: {sorted(set(duplicates))}")
    return dataset


def main():
    # dataset은 product의 list
    # product 하나는 dictionary
    # key: product_name, price, images, reviews.
    with open("dataset.json", "r") as f:
        dataset = json.load(f)

    dataset = validate_image_sequence(dataset)
    


if __name__ == "__main__":
    main()