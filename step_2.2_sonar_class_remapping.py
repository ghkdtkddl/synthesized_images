import os

# YOLO 라벨 파일 경로
label_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\aihub_JWD_SBD_2nd_modify\valid\labels"

# 클래스 번호 재매핑 정의: {기존번호: 새로운번호}, 삭제할 클래스는 포함하지 않음
class_map = {
    '0': '0',   # ank
    '1': '1',   # artificial reef
    '2': '2',   # etc
    '4': '3',   # fish trap → 3
    '6': '4',   # tire → 4
}
# 삭제할 클래스
classes_to_remove = {'3', '5'}  # fish net, pipe

# 디렉토리 내 .txt 파일 순회
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
        cls = parts[0]
        if cls in classes_to_remove:
            continue  # 삭제할 클래스는 건너뜀
        elif cls in class_map:
            new_cls = class_map[cls]
            new_line = ' '.join([new_cls] + parts[1:]) + '\n'
            new_lines.append(new_line)

    # 파일 덮어쓰기
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"✔ 클래스 리맵 완료: {filename}")

print("\n✅ 전체 라벨 클래스 번호 재매핑 및 삭제 완료.")
