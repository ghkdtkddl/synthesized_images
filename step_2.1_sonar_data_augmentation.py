
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import glob

# 90도 회전 (Rotation by 90 degrees)
def rotate_image_90(image, angle, labels):
    h, w = image.shape[:2]
    if angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        new_labels = [[cls, 1-y, x, h, w] for cls, x, y, w, h in labels]
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        new_labels = [[cls, 1-x, 1-y, w, h] for cls, x, y, w, h in labels]
    elif angle == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_labels = [[cls, y, 1-x, h, w] for cls, x, y, w, h in labels]
    else:
        rotated = image
        new_labels = labels
    return rotated, new_labels

# 수평 뒤집기 (Horizontal Flip)
def horizontal_flip(image, labels):
    flipped = cv2.flip(image, 1)
    new_labels = [[cls, 1-x, y, w, h] for cls, x, y, w, h in labels]
    return flipped, new_labels

# 수직 뒤집기 (Vertical Flip)
def vertical_flip(image, labels):
    flipped = cv2.flip(image, 0)
    new_labels = [[cls, x, 1-y, w, h] for cls, x, y, w, h in labels]
    return flipped, new_labels

# 노이즈 추가 (Adding Noise)
def add_speckle_noise(image,labels, noise_level=0.5):
    row, col,ch = image.shape
    speckle_noise = np.random.randn(row, col,ch).astype(np.float32) * noise_level
    noisy_image = image.astype(np.float32) + image.astype(np.float32) * speckle_noise
    noisy = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy , labels

# Mosaic (Combine four images into one)
def mosaic(images, labels, output_size=640):
    # Assume all input images are the same size
    h, w, ch = images[0].shape
    mosaic_h, mosaic_w = 2 * h, 2 * w

    mos_image = np.zeros((mosaic_h, mosaic_w, ch), dtype=images[0].dtype)
    mos_labels = []

    positions = [(0, 0), (0, w), (h, 0), (h, w)]
    for i, (image, lbls) in enumerate(zip(images, labels)):
        y, x = positions[i]
        mos_image[y:y+h, x:x+w] = image
        for label in lbls:
            cls, xc, yc, bw, bh = label
            new_xc = (x + xc * w) / mosaic_w
            new_yc = (y + yc * h) / mosaic_h
            new_bw = bw / 2
            new_bh = bh / 2
            mos_labels.append([cls, new_xc, new_yc, new_bw, new_bh])

    resized_mos_image = cv2.resize(mos_image, (output_size, output_size))
    scale_x = output_size / mosaic_w
    scale_y = output_size / mosaic_h
    # resized_mos_labels = [[cls, xc * scale_x, yc * scale_y, bw * scale_x, bh * scale_y] for cls, xc, yc, bw, bh in mos_labels]

    return resized_mos_image, mos_labels

# 라벨 파일 읽기 (YOLOv5 형식)
def read_labels(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float (parts[4])
            labels.append([cls, x_center, y_center, width, height])
    return labels

# 라벨 파일 저장 (YOLOv5 형식)
def save_labels(label_path, labels):
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

# 소나 이미지 및 라벨 불러오기 (RGB로 불러옴)
def load_image_and_labels(image_path, label_path):
    # image = cv2.imread(image_path)
    img_array = np.fromfile(image_path, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    labels = read_labels(label_path)
    return image, labels

# 파일 경로 설정 (필요에 따라 수정)
# image_path = r'D:\sonar_image\data_augment\test\artificial reef group_20090307_002_22035_12.jpg'
# label_path = r'D:\sonar_image\data_augment\test\artificial reef group_20090307_002_22035_12.txt'
# image, labels = load_image_and_labels(image_path, label_path)

image_path = glob.glob(r'D:\2024_sonar_detection\dataset\final_dataset\synthesized_data\images\*.jpg')
label_path = glob.glob(r'D:\2024_sonar_detection\dataset\final_dataset\synthesized_data\labels\*.txt')

for image_path, label_path in zip(image_path,label_path):
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    image, labels = load_image_and_labels(image_path, label_path)
    # print(image,labels)
    # cv2.imshow(image)
    
    # 데이터 증강 적용
    rotated_image_90, rotated_labels_90 = rotate_image_90(image, 90, labels)
    rotated_image_180, rotated_labels_180 = rotate_image_90(image, 180, labels)
    rotated_image_270, rotated_labels_270 = rotate_image_90(image, 270, labels)
    h_flipped_image, h_flipped_labels = horizontal_flip(image, labels)
    v_flipped_image, v_flipped_labels = vertical_flip(image, labels)
    noisy_image, noisy_labels = add_speckle_noise(image, labels)
    # Mosaic 적용 (회전된 이미지들과 원본 이미지를 사용)
    mosaic_image, mosaic_labels = mosaic([image, rotated_image_90, rotated_image_180, rotated_image_270],
                                        [labels, rotated_labels_90, rotated_labels_180, rotated_labels_270])

    # 이미지 저장 경로 설정
    images_output_dir = r'D:\2024_sonar_detection\dataset\final_dataset\synthesized_data\images_augment'
    labels_output_dir = r'D:\2024_sonar_detection\dataset\final_dataset\synthesized_data\labels_augment'
    
    os.makedirs(images_output_dir,exist_ok=True)
    os.makedirs(labels_output_dir,exist_ok=True)
    
        # 이미지와 라벨 저장
    if not cv2.imwrite(f'{images_output_dir}/{file_name}.jpg', image):
        print(f'Faild to save image:{file_name}.jpg')
    save_labels(f'{labels_output_dir}/{file_name}.txt', labels)
    
    cv2.imwrite(f'{images_output_dir}/{file_name}_rotated_90.jpg', rotated_image_90)
    save_labels(f'{labels_output_dir}/{file_name}_rotated_90.txt', rotated_labels_90)

    cv2.imwrite(f'{images_output_dir}/{file_name}_rotated_180.jpg', rotated_image_180)
    save_labels(f'{labels_output_dir}/{file_name}_rotated_180.txt', rotated_labels_180)

    cv2.imwrite(f'{images_output_dir}/{file_name}_rotated_270.jpg', rotated_image_270)
    save_labels(f'{labels_output_dir}/{file_name}_rotated_270.txt', rotated_labels_270)

    cv2.imwrite(f'{images_output_dir}/{file_name}_horizontal_flip.jpg', h_flipped_image)
    save_labels(f'{labels_output_dir}/{file_name}_horizontal_flip.txt', h_flipped_labels)

    cv2.imwrite(f'{images_output_dir}/{file_name}_vertical_flip.jpg', v_flipped_image)
    save_labels(f'{labels_output_dir}/{file_name}_vertical_flip.txt', v_flipped_labels)

    # cv2.imwrite(f'{images_output_dir}/{file_name}_noisy.jpg', noisy_image)
    # save_labels(f'{labels_output_dir}/{file_name}_noisy.txt', noisy_labels)

    cv2.imwrite(f'{images_output_dir}/{file_name}_mosaic.jpg', mosaic_image)
    save_labels(f'{labels_output_dir}/{file_name}_mosaic.txt', mosaic_labels)

    # 라벨 시각화 함수
#     def plot_labels(image, labels, ax, title='Image'):
#         h, w = image.shape[:2]
#         ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         for label in labels:
#             cls, x, y, width, height = label
#             x1 = int((x - width / 2) * w)
#             y1 = int((y - height / 2) * h)
#             x2 = int((x + width / 2) * w)
#             y2 = int((y + height / 2) * h)
#             rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none')
#             ax.add_patch(rect)
#         ax.set_title(title)

# # 원본 이미지와 증강된 이미지 시각화
#     fig, ax = plt.subplots(2, 4, figsize=(20, 15))

#     plot_labels(image, labels, ax[0, 0], 'Original Image')
#     plot_labels(rotated_image_90, rotated_labels_90, ax[0, 1], 'Rotated 90 degrees')
#     plot_labels(rotated_image_180, rotated_labels_180, ax[0, 2], 'Rotated 180 degrees')
#     plot_labels(rotated_image_270, rotated_labels_270, ax[0, 3], 'Rotated 270 degrees')
#     plot_labels(mosaic_image, mosaic_labels, ax[1, 0], 'Mosaic Image')
#     plot_labels(h_flipped_image, h_flipped_labels, ax[1, 1], 'Horizontal Flip')
#     plot_labels(v_flipped_image, v_flipped_labels, ax[1, 2], 'Vertical Flip')
#     plot_labels(noisy_image, noisy_labels, ax[1, 3], 'Noisy Image')

#     plt.tight_layout()
#     plt.show()

#     # 라벨 확인
#     print("Original Labels:", labels)
#     print("Rotated 90 Labels:", rotated_labels_90)
#     print("Rotated 180 Labels:", rotated_labels_180)
#     print("Rotated 270 Labels:", rotated_labels_270)
#     print("Mosaic Labels:", mosaic_labels)
#     print("Horizontal Flip Labels:", h_flipped_labels)
#     print("Vertical Flip Labels:", v_flipped_labels)
#     print("Noisy Labels:", noisy_labels)
