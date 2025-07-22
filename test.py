import os
import shutil
from pathlib import Path
import cv2

# 기본 경로 설정
base_path = Path(r"D:\2024_sonar_detection\sonar_synthetic_image\labeling_tool\mask_output_result_tool")
labeling_result_dir = base_path / "synthesized_labeling_result"
image_source_dir = base_path / "synthesized_result"
label_source_dir = base_path / "synthesized_labels_result"
image_final_dir = base_path / "synthesized_image_final"
label_final_dir = base_path / "synthesized_labels_final"

# 출력 폴더 생성
image_final_dir.mkdir(parents=True, exist_ok=True)
label_final_dir.mkdir(parents=True, exist_ok=True)

# 기준 파일 목록 (확장자 제거)
labeling_basenames = {f.stem for f in labeling_result_dir.glob("*") if f.is_file()}

copied = 0
skipped = 0

for basename in labeling_basenames:
    img_file = image_source_dir / f"{basename}.jpg"
    label_file = label_source_dir / f"{basename}.txt"

    if img_file.exists() and label_file.exists():
        # PNG → JPG 변환 후 저장
        image = cv2.imread(str(img_file))
        jpg_path = image_final_dir / f"{basename}.jpg"
        cv2.imwrite(str(jpg_path), image)

        # TXT 파일 복사
        shutil.copy(label_file, label_final_dir / label_file.name)

        print(f"✔ 복사됨: {jpg_path.name}, {label_file.name}")
        copied += 1
    else:
        print(f"⚠ 생략됨 (파일 없음): {basename}")
        skipped += 1

print(f"\n✅ 총 복사된 항목: {copied}")
print(f"⚠ 생략된 항목: {skipped}")
