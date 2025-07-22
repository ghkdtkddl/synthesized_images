import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 경로 설정
images_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\aihub_JWD_SBD_2nd_modify\train\images_change"
labels_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\aihub_JWD_SBD_2nd_modify\train\labels_change"
output_dir = r"D:\2024_sonar_detection\dataset\final_dataset\nas_data\aihub_JWD_SBD_2nd_modify\train\output"

# 출력 폴더 생성
os.makedirs(output_dir, exist_ok=True)

# 클래스 리스트
class_list = {
     
    # 'ank' : 0,
    # 'artificial reef': 1,
    # 'etc' : 2,
    # 'fish_net' : 3,
    # 'fish_trap' :4,
    # 'pipe' : 5,
    # 'tire' : 6
    'ank' : 0,
    'artificial reef': 1,
    'etc' : 2,
    'fish_trap' : 3,
    'tire' :4
    
}


def extract_name(dictionary, class_num):  
    for name, num in dictionary.items():
        if num == class_num:
            return name
    return None

# 모든 이미지 처리
for filename in os.listdir(images_dir):
    if filename.endswith(".jpg"):
        name = os.path.splitext(filename)[0]

        # 이미지 읽기
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, f"{name}.txt")
        output_path = os.path.join(output_dir, f"{name}.jpg")

        if not os.path.exists(label_path):
            print(f"Label file not found for {name}, skipping...")
            continue

        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 바운딩 박스 읽기
        bboxes = np.loadtxt(fname=label_path, delimiter=' ', ndmin=2)
        label = np.roll(bboxes, 4, axis=1).tolist()

        # Matplotlib 시각화
        fig, ax = plt.subplots()
        ax.imshow(img)

        for i in range(len(label)):
            dw = img.shape[1]
            dh = img.shape[0]

            # 바운딩 박스 계산
            x1 = (label[i][0] - label[i][2] / 2) * dw
            y1 = (label[i][1] - label[i][3] / 2) * dh
            w = label[i][2] * dw
            h = label[i][3] * dh

            # 바운딩 박스 추가
            rect = patches.Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # 클래스 이름 추가 (옵션)
            class_id = int(label[i][-1])
            class_name = extract_name(class_list, class_id)
            if class_name:
                ax.text(x1, y1 - 10, class_name, fontsize=10, color='yellow', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

        plt.axis('off')

        # 이미지 저장
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Processed and saved: {output_path}")

print("Processing complete!")

print("이미지 폴더:", images_dir)
print("라벨 폴더:", labels_dir)
print("출력 폴더:", output_dir)
print("이미지 개수:", len(os.listdir(images_dir)))
