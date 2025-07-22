import os
import shutil

# 경로 설정
images_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\JWD+SBD_ori_data\train\images"
etc_label_jpg_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\JWD+SBD_ori_data\train\raw_etc_labeling_output"
output_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\JWD+SBD_ori_data\train\raw_etc_output"

# 출력 폴더 생성
os.makedirs(output_dir, exist_ok=True)

# 라벨링 결과 폴더에 있는 jpg 파일명 추출
label_image_names = set(f for f in os.listdir(etc_label_jpg_dir) if f.endswith(".jpg"))

# 이미지 폴더 내에서 일치하는 파일만 복사
count = 0
for filename in os.listdir(images_dir):
    if filename.endswith(".jpg") and filename in label_image_names:
        src_path = os.path.join(images_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        shutil.copy2(src_path, dst_path)
        print(f"✅ 복사됨: {filename}")
        count += 1

print(f"\n총 {count}개의 파일이 raw_etc_output으로 복사 완료되었습니다.")
