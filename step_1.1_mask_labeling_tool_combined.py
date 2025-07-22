import sys
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QListWidget, QGroupBox, QCheckBox, QSlider, QGridLayout, QShortcut, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtCore import Qt, QPoint
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch

class ZoomableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.0
        self.image = None

    def set_image(self, image):
        self.image = image
        self.update_display()

    def wheelEvent(self, event):
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale_factor *= 1.1
            else:
                self.scale_factor /= 1.1
            self.update_display()

    def update_display(self):
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        resized = cv2.resize(self.image, (int(width * self.scale_factor), int(height * self.scale_factor)))
        qimg = QImage(resized.data, resized.shape[1], resized.shape[0], resized.strides[0], QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

class MaskEditor(QMainWindow):
    def __init__(self, project_dir):
        super().__init__()
        self.setWindowTitle("YOLO 기반 통합 라벨링 툴")
        self.setFixedSize(1600, 900)

        self.project_dir = Path(project_dir)
        self.image_dir = self.project_dir / "images"
        self.mask_dir = self.project_dir / "mask"
        self.rgb_dir = self.project_dir / "RGB"
        self.label_list_path = self.project_dir / "labeling_list.json"

        self.yolo_model = YOLO("sonar_detection_yolov10.pt")

        self.mask_dir.mkdir(exist_ok=True)
        self.rgb_dir.mkdir(exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(self.device)
        self.predictor = SamPredictor(self.sam)

        self.image = None
        self.mask = None
        self.image_path = None
        self.mask_path = None
        self.rgb_path = None

        self.brush_size = 10
        self.drawing = False
        self.erasing = False
        self.mask_history = []
        self.cursor_pos = None
        self.rgb_overlay_enabled = False
        self.current_rgb_overlay = None

        self.label_data = []

        self.init_ui()
        self.populate_image_list()
        self.populate_rgb_list()
        self.populate_mask_list()
        
        
    def init_ui(self):
        self.image_label = ZoomableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.draw_button = QPushButton("✏️ 덧칠")
        self.erase_button = QPushButton("🧽 지우기")
        self.draw_button.setCheckable(True)
        self.erase_button.setCheckable(True)
        self.draw_button.setChecked(True)
        self.draw_button.clicked.connect(lambda: self.set_mode(False))
        self.erase_button.clicked.connect(lambda: self.set_mode(True))

        self.rgb_overlay_checkbox = QCheckBox("RGB 오버레이")
        self.rgb_overlay_checkbox.stateChanged.connect(self.toggle_rgb_overlay)

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(20)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.setFixedWidth(80)
        self.brush_slider.valueChanged.connect(self.change_brush_size)

        self.brush_label = QLabel(f"브러시: {self.brush_size}")
        self.undo_button = QPushButton("↩️ Undo")
        self.undo_button.clicked.connect(self.undo)
        self.save_button = QPushButton("💾 저장")
        self.save_button.clicked.connect(self.save_mask)

        self.sam_button = QPushButton("🎯 SAM 마스크 추출")
        self.sam_button.clicked.connect(self.apply_sam)

        QShortcut(QKeySequence("Ctrl+S"), self, self.save_mask)
        QShortcut(QKeySequence("="), self, lambda: self.change_brush_size(self.brush_size + 1))
        QShortcut(QKeySequence("-"), self, lambda: self.change_brush_size(self.brush_size - 1))
        QShortcut(QKeySequence("D"), self, lambda: self.set_mode(False))
        QShortcut(QKeySequence("E"), self, lambda: self.set_mode(True))

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.draw_button)
        top_layout.addWidget(self.erase_button)
        top_layout.addWidget(self.brush_label)
        top_layout.addWidget(self.brush_slider)
        top_layout.addWidget(self.undo_button)
        top_layout.addWidget(self.save_button)
        top_layout.addWidget(self.sam_button)
        top_layout.addWidget(self.rgb_overlay_checkbox)
        top_layout.addStretch()

        # UI에 생성할 것 추가
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_image_from_list)

        self.rgb_list = QListWidget()
        self.rgb_list.itemClicked.connect(self.load_rgb_from_list)

        self.mask_list = QListWidget()
        self.mask_list.itemClicked.connect(self.load_binary_mask_from_list)
        
        image_group = QGroupBox("🖼 이미지 목록")
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_list)
        image_group.setLayout(image_layout)

        rgb_group = QGroupBox("🌈 RGB 마스크 목록")
        rgb_layout = QVBoxLayout()
        rgb_layout.addWidget(self.rgb_list)
        rgb_group.setLayout(rgb_layout)

        mask_group = QGroupBox("⚫ 이진 마스크 목록")
        mask_layout = QVBoxLayout()
        mask_layout.addWidget(self.mask_list)
        mask_group.setLayout(mask_layout)
        
        # 위치 추가
        grid_layout = QGridLayout()
        grid_layout.addWidget(image_group, 0, 0)
        grid_layout.addWidget(rgb_group, 1, 0)
        grid_layout.addWidget(mask_group,2,0)
        right_panel = QWidget()
        right_panel.setLayout(grid_layout)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        body_layout = QHBoxLayout()
        body_layout.addWidget(self.image_label, 4)
        body_layout.addWidget(right_panel, 1)
        main_layout.addLayout(body_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def populate_image_list(self):
        self.image_list.clear()
        for f in sorted(self.image_dir.glob("*.png")) + sorted(self.image_dir.glob("*.jpg")):
            self.image_list.addItem(f.name)

    def populate_rgb_list(self):
        self.rgb_list.clear()
        for f in sorted(self.rgb_dir.glob("*.png")):
            self.rgb_list.addItem(f.name)

    def populate_mask_list(self):
        self.mask_list.clear()
        for f in sorted(self.mask_dir.glob("*.png")):
            self.mask_list.addItem(f.name)

    def load_image_from_list(self, item):
        file_path = self.image_dir / item.text()
        self.image_path = str(file_path)
        image = cv2.imread(self.image_path)
        height, width = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.yolo_model.predict(image, verbose=False)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)

        self.mask = np.zeros((height, width), dtype=np.uint8)
        self.current_rgb_overlay = rgb_image.copy()
        self.current_rgb_overlay[:] = 0

        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(self.mask, (x1, y1), (x2, y2), 255, -1)
            self.current_rgb_overlay[self.mask == 255] = (255, 255, 255)

        self.image = rgb_image
        self.mask_history.clear()
        self.update_display()

    def load_rgb_from_list(self, item):
        file_path = self.rgb_dir / item.text()
        if file_path.exists():
            rgb = cv2.imread(str(file_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            self.image = rgb.copy()
            self.current_rgb_overlay = rgb.copy()
            self.mask = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            self.mask[self.mask > 0] = 255
            self.image_path = self.image_dir / item.text().replace("_rgb_mask.png", ".png")
            self.update_display()
    def load_binary_mask_from_list(self, item):
        file_path = self.mask_dir / item.text()
        if file_path.exists():
            mask = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            self.mask = mask.copy()
            self.image = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # 시각화용
            self.current_rgb_overlay = self.image.copy()
            self.image_path = self.image_dir / item.text().replace("_mask.png", ".png")
            self.update_display()
            
    def update_display(self):
        if self.image is None or self.mask is None:
            return

        base = self.image.copy()
        if self.rgb_overlay_enabled and self.current_rgb_overlay is not None:
            base = cv2.addWeighted(base, 0.7, self.current_rgb_overlay, 0.3, 0)

        mask_rgb = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(base, 0.8, mask_rgb, 0.2, 0)

        if self.cursor_pos:
            cv2.circle(overlay, self.cursor_pos, self.brush_size, (255, 255, 255), 1)

        self.image_label.set_image(overlay)
        self.brush_label.setText(f"브러시: {self.brush_size}")

    def save_mask(self):
        if self.image_path is None or self.mask is None:
            return

        image_name = Path(self.image_path).stem
        rgb_file = self.rgb_dir / f"{image_name}_rgb_mask.png"
        binary_mask_file = self.mask_dir / f"{image_name}_binary_mask.png"  
        if rgb_file.exists():
            QMessageBox.warning(self, "중복 경고", f"{rgb_file.name} 파일이 이미 존재합니다.")
            return
        
        #RGB 마스크 저장
        rgb = self.image.copy()
        rgb[self.mask == 0] = 0
        Image.fromarray(rgb).save(rgb_file)

        #이진 마스크 저장(단일 채널)
        cv2.imwrite(str(binary_mask_file), self.mask)
        
        self.statusBar().showMessage("✅ RGB 및 이진 마스크 저장 완료", 2000)
        self.populate_rgb_list()
        self.populate_mask_list()
        
    def apply_sam(self):
        if self.image is None:
            return

        self.predictor.set_image(self.image)
        masks_accumulated = np.zeros(self.image.shape[:2], dtype=np.uint8)

        results = self.yolo_model.predict(self.image, verbose=False)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)

        for box in boxes:
            x1, y1, x2, y2 = box
            input_box = np.array([x1, y1, x2, y2])
            masks, _, _ = self.predictor.predict(
                box=input_box[None, :],
                multimask_output=True
            )
            masks_accumulated = np.logical_or(masks_accumulated, masks[0])

        self.mask = (masks_accumulated * 255).astype(np.uint8)
        self.current_rgb_overlay = self.image.copy()
        self.current_rgb_overlay[self.mask == 0] = 0
        self.update_display()

    def set_mode(self, erasing):
        self.erasing = erasing
        self.draw_button.setChecked(not erasing)
        self.erase_button.setChecked(erasing)
        self.update_display()

    def change_brush_size(self, value):
        self.brush_size = max(1, min(20, value))
        self.brush_slider.setValue(self.brush_size)
        self.update_display()

    def toggle_rgb_overlay(self, state):
        self.rgb_overlay_enabled = state == Qt.Checked
        self.update_display()

    def push_history(self):
        self.mask_history.append(self.mask.copy())
        if len(self.mask_history) > 10:
            self.mask_history.pop(0)

    def undo(self):
        if self.mask_history:
            self.mask = self.mask_history.pop()
            self.update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.cursor_pos = self.translate_pos(event.pos())
            self.push_history()
            self.draw(self.cursor_pos)

    def mouseMoveEvent(self, event):
        self.cursor_pos = self.translate_pos(event.pos())
        if self.drawing:
            self.draw(self.cursor_pos)
        self.update_display()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def translate_pos(self, pos):
        label_pos = self.image_label.mapFrom(self, pos)
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return None
        pixmap_size = pixmap.size()
        label_size = self.image_label.size()
        offset_x = (label_size.width() - pixmap_size.width()) // 2
        offset_y = (label_size.height() - pixmap_size.height()) // 2
        x = (label_pos.x() - offset_x) * self.mask.shape[1] // pixmap_size.width()
        y = (label_pos.y() - offset_y) * self.mask.shape[0] // pixmap_size.height()
        return (x, y)

    def draw(self, pos):
        if self.image is None or self.mask is None or pos is None:
            return
        x, y = pos
        if 0 <= x < self.mask.shape[1] and 0 <= y < self.mask.shape[0]:
            color = 0 if self.erasing else 255
            cv2.circle(self.mask, (x, y), self.brush_size, color, -1)
            if not self.erasing:
                self.current_rgb_overlay[self.mask == 255] = (255, 255, 255)
            else:
                self.current_rgb_overlay[self.mask == 0] = (0, 0, 0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python mask_labeling_tool_combined.py [프로젝트_경로]")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MaskEditor(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
