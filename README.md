경로 : d:\2024_sonar_detection\sonar_synthetic_image\labeling_tool\mask_output_result_tool

1. 합성이미지 구축
1-1. step_1.1_mask_labeling_tool_combined.py 살행
  - images와 background 폴더에 데이터 추가
  - 라벨링 툴 실행 후 저장 클릭 시, 객체 분리하여 RGB Mask, Mask 생성
1-2. step_1.2_make_synthesized_images.py 실행
  - RGB Mask와 background 데이터에서 랜덤으로 1~3개 객체를 배경에 합성이미지 추가
  - synthesized_result,synthesized_labels_result, synthesized_labeling_result에 저장
  - synthesized_result는 합성이미지 원본, synthesized_labeling_result는 합성이미지의 객체 바운딩박스 좌표, synthesized_labeling_result는 합성이미지와 바운딩박스의 결과물을 시각화 한것임
1-3. Roboflow에서 재라벨링 수행
  - 최종 결과물은 synthesized_image_final, synthesized_labels_final로 저장됨
  - labels값이 0으로 되어있기 때문에 합성이미지 객체에 맞는 class 번호로 수정해야함(step_1.3_roboflow_labels_number_modify.py실행)
  - 파일명이 수정되기 떄문에 기존의 파일명 + roboflow 임의의 파일로 수정되어 일관되게 수정해야함(step_1.4_roboflow_to_yolo_data.py)

2. 합성이미지 데이터 증강 및 클래스 수정
2.1 step_2.1_sonar_data_augmentation.py 실행
  - rotation(90,180,270), flip(수직, 수평), mosaic
2.2 step_2.2_sonar_class_remapping.py 실행
  - 클래스 수정. 및 해당하지 않는 클래스는 삭제
2.3 step_2.3_each_class_instance_check.py 실행
  - labels 파일로 각 클래스 인스턴스 확인.