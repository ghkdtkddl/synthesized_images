# 특정 클래스만을 옮기는 코드

import os
import shutil

# 경로 정의
label_src_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\aihub_JWD_SBD_2nd_modify\train\labels"
image_src_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\aihub_JWD_SBD_2nd_modify\train\images"
label_dst_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\aihub_JWD_SBD_2nd_modify\train\labels_modify"
image_dst_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\aihub_JWD_SBD_2nd_modify\train\images_modifiy"

# 출력 폴더 없으면 생성
os.makedirs(label_dst_dir, exist_ok=True)
os.makedirs(image_dst_dir, exist_ok=True)

# 타겟 클래스
target_classes = {'3', '5'}

# 라벨 파일 순회
for label_file in os.listdir(label_src_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(label_src_dir, label_file)
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 클래스 3 또는 5 포함 여부
    if any(line.strip().split()[0] in target_classes for line in lines):
        # 라벨 파일 이동
        shutil.move(label_path, os.path.join(label_dst_dir, label_file))
        print(f"✔ 라벨 이동됨: {label_file}")

        # 이미지 파일 이름
        base_name = os.path.splitext(label_file)[0]
        image_filename = base_name + ".jpg"
        image_path = os.path.join(image_src_dir, image_filename)

        if os.path.exists(image_path):
            shutil.move(image_path, os.path.join(image_dst_dir, image_filename))
            print(f"✔ 이미지 이동됨: {image_filename}")
        else:
            print(f"⚠ 이미지 없음 (건너뜀): {image_filename}")

print("\n✅ 클래스 3 또는 5 포함된 라벨 + 이미지 이동 완료.")
