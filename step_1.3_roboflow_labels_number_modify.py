import os

# YOLO 라벨 디렉토리
label_dir = r"D:\2024_sonar_detection\sonar_synthetic_image\labeling_tool\mask_output_result_tool\train\synthesized_data\labels"

# 파일 순회
for filename in os.listdir(label_dir):
    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(label_dir, filename)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == '0':
            parts[0] = '3'  # 클래스 0 → 3 (fish trap)
        new_lines.append(' '.join(parts) + '\n')

    # 덮어쓰기 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"✔ 클래스 0 → 3 변경 완료: {filename}")

print("\n✅ 모든 클래스 0을 클래스 3 (fish trap)으로 수정 완료.")
