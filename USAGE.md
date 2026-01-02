# Model 학습 및 테스트 스크립트 사용 가이드

이 폴더에는 YOLOv11 OBB 학습, DINOv2 학습, 그리고 YOLO + DINOv2 통합 파이프라인 스크립트가 포함되어 있습니다.

---

## 📦 환경 설정

### 1. 가상환경 생성 및 활성화

```bash
cd /home/ciw/work/model

# # 가상환경 생성 (한 번만) <- 가상환경은 이미 생성해둠
# python3 -m venv venv

# 가상환경 활성화 (매번 실행 전에)
source venv/bin/activate
```

<!-- (가상환경에 이미 설치되어있음)
### 2. 패키지 설치

```bash
# requirements.txt 기반으로 모든 패키지 설치
pip install -r requirements.txt
```

**필요한 패키지:**
- `ultralytics` - YOLOv11 학습용
- `torch`, `torchvision` - PyTorch
- `Pillow`, `opencv-python` - 이미지 처리
- `pyyaml`, `tqdm`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `pandas`, `scikit-learn` - 유틸리티 -->

---

## 📋 스크립트 분류

이 폴더의 스크립트는 용도에 따라 다음과 같이 분류됩니다:

### 🎓 학습용 스크립트

모델 학습을 수행하는 스크립트입니다. 학습 데이터를 사용하여 모델을 학습시키고, 테스트 데이터로 성능을 평가합니다. 
테스트: 

**`yolov11_obb.py`** - YOLOv11 OBB/Detection 학습
   - YOLOv11 모델을 사용한 객체 검출 학습
   - OBB(Oriented Bounding Box) 또는 일반 Detection 모드 지원
   - 하이퍼파라미터 튜닝 기능 포함

**`tsne_dino.py`** - DINOv2 분류 학습
   - DINOv2 모델을 사용한 양품/불량 분류 학습
   - 2-class, 4-class, 5-class 모드 지원
   - t-SNE 피처맵 시각화 포함
   - 학습 완료 후 자동으로 test set 평가

### 🧪 테스트용 스크립트

학습된 모델을 사용하여 실제 데이터에 대한 추론 및 평가를 수행하는 스크립트입니다.
**아직 yolo, dino 통합 split 코드를 제작하지 못하여 학습용, 테스트용 데이터셋을 따로 둬야 함**

train,val,test 로 split해둔 test데이터셋에 대해 테스트
**`yolov11_obb.py`**
   - `--test-best` 옵션으로 학습된 모델의 test set 평가
   - OBB/Detection 모드 모두 지원
   - Confusion Matrix 및 mAP 메트릭 생성
   - 결과는 `{학습결과폴더}/test_results/`에 저장

**`tsne_dino.py`** 
   - 학습 완료 후 자동으로 테스트 수행

split해두지 않은 데이터셋에 대해 테스트
**`yolov11_dinov2.py`** - YOLO + DINOv2 통합 테스트 파이프라인
   - YOLO로 객체 검출 후 DINOv2로 분류하는 2단계 파이프라인
   - Frontdoor/Bolt 모드 지원
   - Hard/Soft Voting 방식 지원
   - OBB 모드 지원
   - Confusion Matrix 및 성능 메트릭 자동 생성

### 🛠️ 유틸리티 모듈

학습/테스트 과정을 지원하는 보조 기능을 제공하는 스크립트입니다.

**`restore_bak_files.py`** - 라벨 파일 복구 유틸리티
   - YOLO OBB 학습 중 변환된 라벨 파일(.txt.bak)을 원본으로 복구
   - `yolov11_obb.py`의 `--convert-format` 사용 시 자동 복원

---

## 🚀 스크립트 사용법

### 1. yolov11_obb.py - YOLOv11 OBB 학습 (학습용 + 테스트 기능)

YOLOv11 모델을 사용하여 OBB(Oriented Bounding Box) 또는 일반 Detection 학습을 수행하는 스크립트입니다.

**카테고리:** 🎓 학습용 스크립트 (test set 평가 기능 포함)

#### 기본 학습

```bash
# OBB 모드 + 라벨 포맷 변환
# 학습하는 동안 yolo 포맷으로 라벨 txt 일시적으로 변환. 학습 종료 혹은 중단 시, 라벨 복구.
python yolov11_obb.py \
    --project Bolt \
    --data-yaml yaml/BoltYOLO.yaml \
    --obb \
    --convert-format

# 일반 Detection 모드 (BB 모드)
python yolov11_obb.py \
    --project Door \
    --data-yaml yaml/DoorYOLO.yaml
```

#### 하이퍼파라미터 튜닝

```bash
# 튜닝 실행
python yolov11_obb.py \
    --project Bolt \
    --data-yaml yaml/BoltYOLO.yaml \
    --obb \
    --convert-format \
    --tune \
    --tune-iterations 30

# 튜닝된 하이퍼파라미터로 학습
python yolov11_obb.py \
    --project Bolt \
    --data-yaml yaml/BoltYOLO.yaml \
    --obb \
    --convert-format \
    --use-tuned runs/Bolt_YYYYMMDD_HHMMSS_tune/best_hyperparameters.yaml
```

#### Test Set 평가

```bash
# best.pt 모델로 test set 평가
python yolov11_obb.py \
    --test-best runs/Bolt_YYYYMMDD_HHMMSS/weights/best.pt \
    --test-data-yaml yaml/BoltYOLO.yaml \
    --obb

# 또는 txt 파일 직접 지정
python yolov11_obb.py \
    --test-best runs/Bolt_YYYYMMDD_HHMMSS/weights/best.pt \
    --test-txt /home/ciw/work/datasets/CODE/TXT/test_Bolt.txt \
    --obb
```

**주요 옵션:**
- `--project`: 프로젝트 이름 (필수)
- `--data-yaml`: 데이터셋 YAML 파일 경로
- `--obb`: OBB 모드 활성화
- `--convert-format`: 라벨 포맷 변환 (xywha → xyxyxyxy)
- `--tune`: 하이퍼파라미터 튜닝 실행
- `--tune-iterations`: 튜닝 반복 횟수 (기본값: 30)
- `--use-tuned`: 튜닝된 하이퍼파라미터 YAML 파일 경로
- `--test-best`: best.pt 모델 경로 (test 평가용)
- `--test-data-yaml`: test set YAML 파일 경로
- `--test-txt`: test 이미지 경로 txt 파일
- `--model`: 모델 타입 (기본값: yolo11s.pt)
- `--epochs`: 학습 에포크 수 (기본값: 70)
- `--batch`: 배치 크기 (기본값: 16)
- `--imgsz`: 이미지 크기 (기본값: 640)
- `--no-cleanup`: 학습 후 .bak 파일 자동 복원 비활성화

**출력:**
- `runs/{project_name}_{timestamp}/`: 학습 결과 폴더
  - `weights/best.pt`: 최적 모델
  - `weights/last.pt`: 마지막 에포크 모델
  - `confusion_matrix.png`: 혼동행렬
  - `results.png`: 학습 곡선

---

### 2. tsne_dino.py - DINOv2 학습 (학습용 + 테스트 기능)

DINOv2 모델을 사용하여 양품/불량 분류 학습을 수행하는 스크립트입니다.

**카테고리:** 🎓 학습용 스크립트 (test set 평가 기능 포함)

#### 볼트 데이터 학습

```bash
# 2-class 모드 (simple)
python tsne_dino.py \
    --project BoltDINO \
    --data-yaml yaml/BoltDINO.yaml \
    --model-size base \
    --imgsz 224 \
    --batch 32 \
    --epochs 70 \
    --lr-backbone 1e-5 \
    --lr-head 1e-4 \
    --freeze-epochs 5

# 4-class 모드
python tsne_dino.py \
    --project BoltDINO_4class \
    --data-yaml yaml/BoltDINO_4class.yaml \
    --model-size base \
    --imgsz 224 \
    --batch 32 \
    --epochs 70 \
    --lr-backbone 1e-5 \
    --lr-head 1e-4 \
    --freeze-epochs 5
```

#### 도어 데이터 학습

```bash
# High 부위 2-class
python tsne_dino.py \
    --project DoorDINO_high_2class \
    --data-yaml yaml/DoorDINO_high_2class.yaml \
    --model-size base \
    --imgsz 224 \
    --batch 32 \
    --epochs 70 \
    --lr-backbone 1e-5 \
    --lr-head 1e-4 \
    --freeze-epochs 5

# High 부위 5-class
python tsne_dino.py \
    --project DoorDINO_high_5class \
    --data-yaml yaml/DoorDINO_high_5class.yaml \
    --model-size base \
    --imgsz 224 \
    --batch 32 \
    --epochs 70 \
    --lr-backbone 1e-5 \
    --lr-head 1e-4 \
    --freeze-epochs 5

# Mid/Low 부위도 동일한 방식으로 실행
```

**주요 옵션:**
- `--project`: 프로젝트 이름 (필수)
- `--data-yaml`: 데이터셋 YAML 파일 경로 (필수)
- `--model-size`: 모델 크기 (`small`, `base`, `large`, `giant`, 기본값: `small`)
- `--imgsz`: 이미지 크기 (기본값: 224)
- `--batch`: 배치 크기 (기본값: 32)
- `--epochs`: 학습 에포크 수 (기본값: 100)
- `--lr`: 학습률 (기본값: 1e-4)
- `--lr-backbone`: 백본 학습률 (기본값: lr의 0.1배)
- `--lr-head`: 분류기 헤드 학습률 (기본값: lr)
- `--freeze-epochs`: 초기 백본 고정 에포크 수 (기본값: 0)
- `--device`: 디바이스 (`cuda` 또는 `cpu`, 기본값: `cuda`)
- `--clean-txt`: 학습 전 txt 파일에서 존재하지 않는 이미지 경로 제거

**출력:**
- `runs/{project_name}_{timestamp}/`: 학습 결과 폴더
  - `weights/best.pt`: 최적 모델 (검증 정확도 기준)
  - `weights/last.pt`: 마지막 에포크 모델
  - `results.png`: 학습 곡선 (Loss, Accuracy)
  - `confusion_matrix.png`: 검증 혼동행렬 (count)
  - `confusion_matrix_normalized.png`: 검증 혼동행렬 (row-normalized)
  - `val_tsne_3d.html`: 3D t-SNE 피처맵 (인터랙티브)
  - `val_tsne_2d.png`: 2D t-SNE 피처맵 (static)
  - `metrics.json`: 학습 메트릭
  - `test_results/`: 테스트 결과 (test_txt가 제공된 경우)
    - `correct/`: 정답 이미지
    - `incorrect/`: 오답 이미지

---

### 3. yolov11_dinov2.py - YOLO + DINOv2 통합 파이프라인 (테스트용)

YOLO로 객체 검출 후 DINOv2로 분류하는 통합 테스트 파이프라인입니다.

**카테고리:** 🧪 테스트용 스크립트

#### 설정 YAML 파일 준비

**frontdoor 모드용 YAML 예시 (`yaml/pipeline_frontdoor.yaml`):**
```yaml
yolo_model: runs/Door_YYYYMMDD_HHMMSS/weights/best.pt
high: runs/DoorDINO_high_2class_YYYYMMDD_HHMMSS/weights/best.pt
mid: runs/DoorDINO_mid_2class_YYYYMMDD_HHMMSS/weights/best.pt
low: runs/DoorDINO_low_2class_YYYYMMDD_HHMMSS/weights/best.pt
```

**bolt 모드용 YAML 예시 (`yaml/pipeline_bolt.yaml`):**
```yaml
yolo_model: runs/Bolt_YYYYMMDD_HHMMSS/weights/best.pt
bolt: runs/BoltDINO_YYYYMMDD_HHMMSS/weights/best.pt
```

#### 실행

```bash
# Frontdoor 모드 (일반 bbox)
python yolov11_dinov2.py \
    --config yaml/pipeline_frontdoor.yaml \
    --txt /home/ciw/work/datasets/CODE/TXT/test_Door.txt \
    --mode frontdoor \
    --voting hard \
    --project frontdoor_test \
    --conf 0.25

# Frontdoor 모드 (OBB)
python yolov11_dinov2.py \
    --config yaml/pipeline_frontdoor.yaml \
    --txt /home/ciw/work/datasets/CODE/TXT/test_Door.txt \
    --mode frontdoor \
    --voting soft \
    --project frontdoor_test_obb \
    --conf 0.25 \
    --obb

# Bolt 모드
python yolov11_dinov2.py \
    --config yaml/pipeline_bolt.yaml \
    --txt /home/ciw/work/datasets/CODE/TXT/test_Bolt.txt \
    --mode bolt \
    --voting hard \
    --project bolt_test \
    --conf 0.25
```

**주요 옵션:**
- `--config`: 모델 경로들이 들어있는 YAML 파일 경로 (필수)
- `--txt`: 처리할 이미지 경로 목록이 담긴 txt 파일 경로 (필수)
- `--mode`: 실행 모드 (`frontdoor`, `door`, `bolt`, 필수)
- `--voting`: 보팅 방식 (`hard` 또는 `soft`, 기본값: `hard`)
- `--project`: 결과 폴더명 prefix (기본값: `pipeline_test`)
- `--conf`: YOLO 신뢰도 임계값 (기본값: 0.25)
- `--device`: 디바이스 (`cuda` 또는 `cpu`, 기본값: `cuda`)
- `--obb`: OBB(Oriented Bounding Box) 모드 사용

**보팅 방식:**
- `hard`: 하나라도 불량이면 불량 (OR 연산)
- `soft`: 평균 불량 confidence가 0.5 이상이면 불량

**출력:**
- `runs/{project_name}_{timestamp}/`: 결과 폴더
  - `crops/`: 크롭된 이미지들
  - `visualizations/`: 시각화 이미지 (바운딩 박스 + 예측 결과)
  - `results.json`: 상세 결과 데이터
  - `confusion_matrix.png`: 혼동행렬 (양품/불량 2-class)
  - `confusion_matrix_normalized.png`: 정규화된 혼동행렬
  - `confusion_matrix_bolt_4class.png`: 볼트 4-class 모드용 (해당하는 경우)
  - `confusion_matrix_door_5class.png`: 도어 5-class 모드용 (해당하는 경우)
  - `metrics.json`: 성능 메트릭 (Accuracy, Precision, Recall, F1)

---

## 📁 YAML 파일 구조

### YOLO 학습용 YAML (`yaml/BoltYOLO.yaml`, `yaml/DoorYOLO.yaml`)

```yaml
train: /home/ciw/work/datasets/CODE/TXT/train_Bolt.txt
val: /home/ciw/work/datasets/CODE/TXT/val_Bolt.txt
test: /home/ciw/work/datasets/CODE/TXT/test_Bolt.txt  # 선택
nc: 2  # 클래스 수
names:
  0: bolt_frontside
  1: bolt_side
```

### DINOv2 학습용 YAML (`yaml/BoltDINO.yaml`, `yaml/DoorDINO_*.yaml`)

```yaml
train: /home/ciw/work/datasets/CODE/TXT/train_dino_Bolt.txt
val: /home/ciw/work/datasets/CODE/TXT/val_dino_Bolt.txt
test: /home/ciw/work/datasets/CODE/TXT/test_dino_Bolt.txt  # 선택
parts: bolt  # 또는 frontdoor
mode: simple  # 2-class 모드 (선택, 없으면 4-class 또는 5-class)
preprocess: on  # 또는 off (선택, 기본값: on)
```

### 파이프라인 설정 YAML (`yaml/pipeline_*.yaml`)

```yaml
# Frontdoor 모드
yolo_model: runs/Door_YYYYMMDD_HHMMSS/weights/best.pt
high: runs/DoorDINO_high_2class_YYYYMMDD_HHMMSS/weights/best.pt
mid: runs/DoorDINO_mid_2class_YYYYMMDD_HHMMSS/weights/best.pt
low: runs/DoorDINO_low_2class_YYYYMMDD_HHMMSS/weights/best.pt

# Bolt 모드
yolo_model: runs/Bolt_YYYYMMDD_HHMMSS/weights/best.pt
bolt: runs/BoltDINO_YYYYMMDD_HHMMSS/weights/best.pt
```

---

## 🔧 주요 기능

### 라벨 포맷 변환 (OBB 모드)

YOLO OBB 학습 시 라벨을 `xywha` 형식에서 `xyxyxyxy` 형식으로 자동 변환합니다.

- 원본 라벨은 `.bak` 파일로 백업
- 학습 완료 후 자동 복원 (Ctrl+C로 중단해도 복원)
- `--no-cleanup` 플래그로 자동 복원 비활성화 가능

### 하이퍼파라미터 튜닝

YOLO 학습 전에 최적 하이퍼파라미터를 자동으로 찾습니다.

- `--tune` 플래그로 튜닝 실행
- `--tune-iterations`로 반복 횟수 조정 (기본값: 30)
- 튜닝 결과는 `best_hyperparameters.yaml`에 저장
- `--use-tuned`로 튜닝된 파라미터 적용

### 클래스 불균형 대응 (DINOv2)

DINOv2 학습 시 Inverse Frequency 기반 클래스 가중치를 자동으로 적용합니다.

- 학습 데이터의 클래스별 빈도에 따라 가중치 계산
- CrossEntropyLoss에 자동 적용

### 백본 고정 학습 (DINOv2)

초기 에포크 동안 백본을 고정하고 헤드만 학습할 수 있습니다.

- `--freeze-epochs`로 고정 에포크 수 지정
- 고정 기간 동안 백본 학습률 = 0
- 고정 해제 후 백본과 헤드 모두 학습

---

## 📊 출력 파일 설명

### YOLO 학습 결과

- `best.pt`: 검증 성능이 가장 좋은 모델
- `last.pt`: 마지막 에포크 모델
- `confusion_matrix.png`: 혼동행렬
- `results.png`: 학습 곡선 (Loss, mAP 등)
- `args.yaml`: 학습 시 사용된 설정

### DINOv2 학습 결과

- `best.pt`: 검증 정확도가 가장 높은 모델
- `last.pt`: 마지막 에포크 모델
- `results.png`: 학습 곡선 (Loss, Accuracy)
- `confusion_matrix.png`: 검증 혼동행렬 (count)
- `confusion_matrix_normalized.png`: 검증 혼동행렬 (row-normalized)
- `val_tsne_3d.html`: 3D t-SNE 피처맵 (인터랙티브, 브라우저에서 열기)
- `val_tsne_2d.png`: 2D t-SNE 피처맵 (static)
- `metrics.json`: 학습 메트릭
- `test_results/`: 테스트 결과 (test_txt 제공 시)
  - `correct/`: 정답 이미지
  - `incorrect/`: 오답 이미지

### 파이프라인 결과

- `crops/`: 크롭된 이미지들
- `visualizations/`: 시각화 이미지 (바운딩 박스 + 예측 결과)
- `results.json`: 상세 결과 데이터
- `confusion_matrix.png`: 혼동행렬 (양품/불량 2-class)
- `confusion_matrix_normalized.png`: 정규화된 혼동행렬
- `metrics.json`: 성능 메트릭

---

## ⚠️ 주의사항

1. **가상환경 활성화**: 모든 스크립트 실행 전에 `source venv/bin/activate` 필수
2. **경로 확인**: YAML 파일의 경로가 현재 서버 환경(`/home/ciw/work/...`)과 일치하는지 확인
3. **OBB 모드**: `--obb` 플래그 사용 시 모델 파일명이 `-obb.pt`로 끝나야 함
4. **라벨 복원**: `--convert-format` 사용 시 학습 완료 후 자동으로 원본 라벨 복원됨
5. **GPU 메모리**: 배치 크기와 이미지 크기에 따라 GPU 메모리 사용량이 달라짐

---

## 🔗 관련 파일

- `datasets/CODE/`: 데이터 전처리 및 split 생성 스크립트
- `yaml/`: 학습 및 파이프라인 설정 YAML 파일들
- `runs/`: 학습 결과 저장 폴더

