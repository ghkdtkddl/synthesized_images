#labels에서 각 객체의 instance를 확인하는 코드

import os
from collections import Counter

# YOLO 라벨 파일들이 들어있는 디렉터리 경로 설정
label_dir = r"/d/2024_sonar_detection/dataset/final_dataset/nas_data/aihub_JWD_2nd/train/labels"

# 클래스 이름 정의 (인덱스와 순서 중요)
class_names = ['ank', 'artificial reef', 'etc', 'fish trap', 'tire']

# 클래스 인스턴스 카운터 초기화
class_counts = Counter()

# 모든 .txt 파일을 순회하며 클래스 카운트
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

# 결과 출력
print("클래스별 인스턴스 수:")
for idx, name in enumerate(class_names):
    print(f"{name} ({idx}): {class_counts.get(idx, 0)}개")
