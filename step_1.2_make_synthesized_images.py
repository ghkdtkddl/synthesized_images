import os
import cv2
import numpy as np
import random

def load_images_from_folder(folder):
    """폴더에서 이미지 파일을 불러와 리스트로 반환"""
    images, filenames = [], []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                filenames.append(fname)
    return images, filenames

def create_mask_from_rgb(rgb_image):
    """객체 RGB 이미지에서 이진 마스크 생성"""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return mask

def soften_mask_edges(mask, iterations=0):
    """경계 부드럽게 마스킹 (객체 선 제거 목적)"""
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=iterations)
    mask = cv2.GaussianBlur(mask, (1, 1), 0)
    return mask

def extract_object_and_mask(rgb_image):
    """객체 이미지에서 객체와 마스크 추출"""
    mask = create_mask_from_rgb(rgb_image)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    obj_crop = rgb_image[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]
    return obj_crop, mask_crop

def rotate_object(obj, mask):
    """객체 회전만 적용 (스케일은 적용하지 않음)"""
    angle = random.uniform(0, 360)
    h, w = obj.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_obj = cv2.warpAffine(obj, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return rotated_obj, rotated_mask

def blend_object_seamless(bg, obj_crop, mask_crop, x, y):
    """SeamlessClone을 사용한 객체 합성"""
    h, w = obj_crop.shape[:2]
    center = (x + w // 2, y + h // 2)
    if center[0] < 0 or center[1] < 0 or center[0] >= bg.shape[1] or center[1] >= bg.shape[0]:
        return bg
    try:
        output = cv2.seamlessClone(obj_crop, bg, mask_crop, center, cv2.NORMAL_CLONE)
    except:
        return bg
    return output

def is_roi_too_dark(bg, x, y, ow, oh, threshold=40):
    """배경 ROI가 너무 어두운 경우 제외"""
    roi = bg[y:y+oh, x:x+ow]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    return mean_val < threshold

def convert_to_yolo_format(x, y, w, h, img_w, img_h):
    """YOLO 형식 (x_center y_center width height 정규화)"""
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def synthesize(rgb_dir, background_dir, output_img_dir, output_label_dir, output_txt_dir, max_objects=3):
    """전체 합성 파이프라인"""
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    rgb_imgs, _ = load_images_from_folder(rgb_dir)
    bg_imgs, bg_fnames = load_images_from_folder(background_dir)

    for idx, (bg, bg_name) in enumerate(zip(bg_imgs, bg_fnames)):
        canvas = bg.copy()
        used_boxes = []
        bh, bw = canvas.shape[:2]
        num_objects = random.randint(1, max_objects)
        yolo_lines = []

        for _ in range(num_objects):
            obj_idx = random.randint(0, len(rgb_imgs) - 1)
            obj_img = rgb_imgs[obj_idx]
            obj_crop, mask_crop = extract_object_and_mask(obj_img)
            obj_crop, mask_crop = rotate_object(obj_crop, mask_crop)
            oh, ow = obj_crop.shape[:2]

            tries = 0
            while tries < 50:
                x = random.randint(0, bw - ow)
                y = random.randint(0, bh - oh)
                overlap = any(abs(x - ux) < ow and abs(y - uy) < oh for (ux, uy, uw, uh) in used_boxes)
                too_dark = is_roi_too_dark(canvas, x, y, ow, oh)
                if not overlap and not too_dark:
                    break
                tries += 1

            soft_mask = soften_mask_edges(mask_crop, iterations=0)
            canvas = blend_object_seamless(canvas, obj_crop, soft_mask, x, y)

            used_boxes.append((x, y, ow, oh))
            x_center, y_center, w_norm, h_norm = convert_to_yolo_format(x, y, ow, oh, bw, bh)
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # ✅ 합성 완료된 이미지 기준으로 라벨링 이미지 생성
        label_canvas = canvas.copy()
        for (x, y, ow, oh) in used_boxes:
            cv2.rectangle(label_canvas, (x, y), (x + ow, y + oh), (0, 255, 0), 2)

        # ✅ 저장
        out_base = os.path.splitext(bg_name)[0] + "_synthesized.jpg"
        cv2.imwrite(os.path.join(output_img_dir, out_base), canvas)
        cv2.imwrite(os.path.join(output_label_dir, out_base), label_canvas)
        with open(os.path.join(output_txt_dir, out_base.replace(".jpg", ".txt")), "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        print(f"✅ Saved: {out_base}")

# ===================== 실행 =====================
if __name__ == "__main__":
    synthesize(
        rgb_dir=r"D:\2024_sonar_detection\sonar_synthetic_image\labeling_tool\mask_output_result_tool\RGB",
        background_dir=r"D:\2024_sonar_detection\sonar_synthetic_image\labeling_tool\mask_output_result_tool\background",
        output_img_dir=r"D:\2024_sonar_detection\sonar_synthetic_image\labeling_tool\mask_output_result_tool\synthesized_result",
        output_label_dir=r"D:\2024_sonar_detection\sonar_synthetic_image\labeling_tool\mask_output_result_tool\synthesized_labeling_result",
        output_txt_dir=r"D:\2024_sonar_detection\sonar_synthetic_image\labeling_tool\mask_output_result_tool\synthesized_labels_result",
        max_objects=3
    )
