import os
import re

# 디렉토리 경로 설정
base_dir = r"C:\Users\jh\Downloads\synthesized_sonar_image.v1i.yolov5pytorch\train"
image_dir = os.path.join(base_dir, "images")
label_dir = os.path.join(base_dir, "labels")

# 파일명에서 "_jpg.rf.<hash>" 패턴을 제거하는 정규표현식
pattern = re.compile(r"(.*)_jpg\.rf\.[a-f0-9]+(\.jpg|\.txt)")

def rename_files_in_dir(target_dir, extension):
    for filename in os.listdir(target_dir):
        if not filename.endswith(extension):
            continue

        match = pattern.match(filename)
        if match:
            new_name = match.group(1) + match.group(2)
            old_path = os.path.join(target_dir, filename)
            new_path = os.path.join(target_dir, new_name)

            # 이름이 충돌하지 않을 때만 리네임
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"✔ Renamed: {filename} → {new_name}")
            else:
                print(f"⚠ Skip (already exists): {new_name}")

# 실행
print("🔧 images 폴더 리네이밍 중...")
rename_files_in_dir(image_dir, ".jpg")

print("\n🔧 labels 폴더 리네이밍 중...")
rename_files_in_dir(label_dir, ".txt")

print("\n✅ 모든 파일 이름 정리가 완료되었습니다.")
