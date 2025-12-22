"""YOLOv11 + DINOv2 테스트 파이프라인"""

import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import yaml
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from math import cos, sin
import math


def compute_rotated_box_corners(cx, cy, w, h, angle):
    """회전된 박스의 4개 모서리 좌표 계산"""
    dx = w / 2.0
    dy = h / 2.0
    
    local_corners = [
        (-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)
    ]
    
    c = cos(angle)
    s = sin(angle)
    
    corners = []
    for lx, ly in local_corners:
        rx = c * lx - s * ly + cx
        ry = s * lx + c * ly + cy
        corners.append((rx, ry))
    
    return corners


def correct_orientation_constrained(w, h, angle):
    """
    [사용자 지정 로직] 형상 적응형 보정 (Shape-Adaptive)
    조건: 객체는 원래 방향(가로/세로)에서 +-45도 이내로만 기울어짐.
    목표: 
      1. w >= h (가로 객체) -> 각도를 오른쪽(-45~+45도)으로 맞춤
      2. h > w (세로 객체) -> 각도를 위쪽(-135~-45도)으로 맞춤
    """
    pi = math.pi
    
    # 1. 각도 1차 정규화 (-pi ~ +pi)
    angle = (angle + pi) % (2 * pi) - pi
    
    # 2. 객체 형태에 따른 방향 보정
    if w >= h:
        # [Case A] 가로가 긴 객체 (Horizontal)
        # 목표: 각도가 0도(오른쪽) 근처여야 함.
        # 만약 각도가 절대값 90도(pi/2)를 넘어가면 '왼쪽'을 보고 있다는 뜻이므로 뒤집음.
        if abs(angle) > pi / 2:
            angle -= pi  # 180도 회전
            
    else:
        # [Case B] 세로가 긴 객체 (Vertical)
        # 목표: 각도가 -90도(위쪽) 근처여야 함. (OpenCV 좌표계: -90도가 12시 방향)
        
        # 간단하게: "Y축 아래(양수 각도)"를 보고 있으면 무조건 위로 올림
        if angle > 0:  
            angle -= pi
        
        # -180도 근처(-pi)인 경우도 아래쪽(6시)에 가까우므로 위로 보냄
        # (단, +-45도 제한 조건 때문에 이 케이스는 드물겠지만 안전장치)
        if angle < -pi + (pi/4): # -135도보다 더 작으면 (예: -170도)
             angle += pi

    # 최종 각도 재정규화
    angle = (angle + pi) % (2 * pi) - pi
            
    return w, h, angle


def correct_orientation_door(w, h, angle, part):
    """
    도어 모드 전용 방향 보정
    - 상단부(high)와 하단부(low): 무조건 세로 (h > w로 강제), 목표 각도 -90도
    - 중단부(mid): 무조건 가로 (w > h로 강제), 목표 각도 0도
    - 모든 부위: 절댓값이 작은 쪽으로 회전 (0에 가까운 쪽)
    
    Args:
        w, h: 너비, 높이
        angle: 회전 각도 (라디안)
        part: 'high', 'mid', 'low'
    
    Returns:
        w, h, angle: 보정된 너비, 높이, 각도
    """
    pi = math.pi
    
    # 1. 각도 정규화 (-pi ~ +pi)
    angle = (angle + pi) % (2 * pi) - pi
    
    # 2. 부위별 강제 방향 적용
    if part in ['high', 'low']:
        # 상단부/하단부: 무조건 세로 (h > w)
        if w > h:
            w, h = h, w  # w와 h 교환
            angle += pi / 2  # 90도 회전
        
        # 목표 각도: -90도 (위쪽)
        target_angle = -pi / 2
        
        # 절댓값이 작은 쪽으로 회전 (0에 가까운 쪽)
        # 현재 각도와 목표 각도의 차이를 계산
        diff = angle - target_angle
        
        # -180~180 범위로 정규화
        diff = (diff + pi) % (2 * pi) - pi
        
        # 절댓값이 90도(pi/2)보다 크면 반대 방향으로 회전
        # 예: diff가 150도면 -30도로, -150도면 30도로
        if abs(diff) > pi / 2:
            diff = diff - pi if diff > 0 else diff + pi
        
        angle = target_angle + diff
        
    elif part == 'mid':
        # 중단부: 무조건 가로 (w > h)
        if h > w:
            w, h = h, w  # w와 h 교환
            angle += pi / 2  # 90도 회전
        
        # 목표 각도: 0도 (오른쪽)
        target_angle = 0.0
        
        # 절댓값이 작은 쪽으로 회전 (0에 가까운 쪽)
        # 현재 각도와 목표 각도의 차이를 계산
        diff = angle - target_angle
        
        # -180~180 범위로 정규화
        diff = (diff + pi) % (2 * pi) - pi
        
        # 절댓값이 90도(pi/2)보다 크면 반대 방향으로 회전
        # 예: diff가 150도면 -30도로, -150도면 30도로
        if abs(diff) > pi / 2:
            diff = diff - pi if diff > 0 else diff + pi
        
        angle = target_angle + diff
    
    # 최종 각도 재정규화
    angle = (angle + pi) % (2 * pi) - pi
    
    return w, h, angle


def point_in_rotated_box(px, py, box_cx, box_cy, box_w, box_h, box_angle):
    """
    점이 회전된 박스 내부에 있는지 확인
    
    Args:
        px, py: 확인할 점의 좌표
        box_cx, box_cy: 박스 중심점
        box_w, box_h: 박스 너비, 높이
        box_angle: 박스 회전 각도 (라디안)
    
    Returns:
        bool: 점이 박스 내부에 있으면 True
    """
    # 점을 박스 중심 기준으로 이동
    dx = px - box_cx
    dy = py - box_cy
    
    # 회전 각도의 역변환 (박스 좌표계로 변환)
    c = cos(-box_angle)
    s = sin(-box_angle)
    
    # 회전된 좌표
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    
    # 박스 내부인지 확인
    return abs(local_x) <= box_w / 2 and abs(local_y) <= box_h / 2


def crop_rotated_object(img, cx, cy, w, h, angle, part=None):
    """
    회전된 객체를 crop
    
    Args:
        img: 이미지
        cx, cy: 중심점
        w, h: 너비, 높이
        angle: 회전 각도 (라디안)
        part: 부위 ('high', 'mid', 'low') - 도어 모드일 때만 사용
    """
    img_h, img_w = img.shape[:2]

    # 방향 보정
    if part is not None and part in ['high', 'mid', 'low']:
        # 도어 모드: 부위별 강제 방향 적용
        w, h, angle = correct_orientation_door(w, h, angle, part)
    else:
        # 일반 모드: 기존 로직 사용
        w, h, angle = correct_orientation_constrained(w, h, angle)
    
    # 각도가 0에 매우 가까우면 일반 crop
    if abs(angle) < 1e-6:
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(img_w, int(cx + w / 2))
        y2 = min(img_h, int(cy + h / 2))
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        crop = img[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
        return crop_resized
    
    src_corners = compute_rotated_box_corners(cx, cy, w, h, angle)
    src_points = np.array(src_corners, dtype=np.float32)
    
    dst_corners = [
        (0, 0), (w, 0), (w, h), (0, h)
    ]
    dst_points = np.array(dst_corners, dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    warped = cv2.warpPerspective(img, M, (int(w), int(h)), 
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
    return warped


class DINOv2Classifier(nn.Module):
    """DINOv2 분류 모델 (체크포인트 로딩용)"""
    def __init__(self, backbone, embed_dim, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class YOLODINOPipeline:
    def __init__(self, mode='frontdoor', yolo_model_path=None, 
                 dino_models=None, device='cuda', conf_threshold=0.25,
                 voting_method='hard', project_name='pipeline_test', use_obb=False):
        """
        YOLO + DINOv2 테스트 파이프라인
        
        Args:
            mode (str): 'frontdoor' 또는 'bolt'
            yolo_model_path (str): YOLO 모델 경로
            dino_models (dict): DINOv2 모델 경로들
                - frontdoor: {'high': path, 'mid': path, 'low': path}
                - bolt: {'bolt': path}
            device (str): 디바이스
            conf_threshold (float): YOLO 신뢰도 임계값
            voting_method (str): 'hard' 또는 'soft' (frontdoor용)
            project_name (str): 프로젝트 이름 (결과 폴더명에 사용)
            use_obb (bool): OBB(Oriented Bounding Box) 모드 사용 여부
        """
        self.mode = mode.lower()
        # 'door'를 'frontdoor'로 정규화
        if self.mode == 'door':
            self.mode = 'frontdoor'
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.voting_method = voting_method
        self.project_name = project_name
        self.use_obb = use_obb
        
        # YOLO 모델 로드
        if yolo_model_path is None:
            raise ValueError("YOLO 모델 경로를 제공해야 합니다.")
        print(f"🔄 YOLO 모델 로드 중: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        
        # DINOv2 모델 로드
        self.dino_models = {}
        self.dino_num_classes = {}  # 각 모델의 클래스 수 저장
        if dino_models is None:
            raise ValueError("DINOv2 모델 경로를 제공해야 합니다.")
        
        if self.mode == 'frontdoor':
            required_keys = ['high', 'mid', 'low']
            for key in required_keys:
                if key not in dino_models:
                    raise ValueError(f"frontdoor 모드는 {required_keys} 모델이 필요합니다.")
            
            for part, model_path in dino_models.items():
                print(f"🔄 DINOv2 모델 로드 중 ({part}): {model_path}")
                model, num_classes = self._load_dino_model(model_path)
                self.dino_models[part] = model
                self.dino_num_classes[part] = num_classes
        
        elif self.mode == 'bolt':
            if 'bolt' not in dino_models:
                raise ValueError("bolt 모드는 'bolt' 모델이 필요합니다.")
            print(f"🔄 DINOv2 모델 로드 중 (bolt): {dino_models['bolt']}")
            model, num_classes = self._load_dino_model(dino_models['bolt'])
            self.dino_models['bolt'] = model
            self.dino_num_classes['bolt'] = num_classes
        
        else:
            raise ValueError(f"지원하지 않는 모드: {self.mode}")
        
        # YOLO 클래스 매핑 (bolt 모드용)
        self.bolt_class_names = {
            0: 'bolt_frontside',
            1: 'bolt_side',
            2: 'sedan (trunklid)',
            3: 'suv (trunklid)',
            4: 'hood',
            5: 'long (frontfender)',
            6: 'mid (frontfender)',
            7: 'short (frontfender)'
        }
        
        # DINOv2 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ 파이프라인 초기화 완료")
        print(f"  - 모드: {self.mode}")
        print(f"  - 디바이스: {self.device}")
        print(f"  - YOLO 신뢰도 임계값: {self.conf_threshold}")
        print(f"  - OBB 모드: {self.use_obb}")
        if self.mode == 'frontdoor':
            print(f"  - Voting 방법: {self.voting_method}")
            # 각 부위별 클래스 수 출력
            for part in ['high', 'mid', 'low']:
                if part in self.dino_num_classes:
                    num_cls = self.dino_num_classes[part]
                    if num_cls == 5:
                        mode_text = "5-class"
                    elif num_cls == 4:
                        mode_text = "4-class"
                    else:
                        mode_text = "2-class (simple)"
                    print(f"  - DINO {part}: {mode_text}")
        else:  # bolt
            print(f"  - Voting 방법: {self.voting_method}")
            num_cls = self.dino_num_classes.get('bolt', 2)
            mode_text = "4-class" if num_cls == 4 else "2-class (simple)"
            print(f"  - DINO bolt: {mode_text}")
    
    def _load_dino_model(self, model_path):
        """DINOv2 모델 체크포인트 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        model_size = config.get('model_size', 'small')
        num_classes = config.get('num_classes', 2)
        
        # 백본 로드
        model_map = {
            'small': ('dinov2_vits14', 384),
            'base': ('dinov2_vitb14', 768),
            'large': ('dinov2_vitl14', 1024),
            'giant': ('dinov2_vitg14', 1536)
        }
        model_name, embed_dim = model_map.get(model_size, ('dinov2_vits14', 384))
        
        backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        model = DINOv2Classifier(backbone, embed_dim, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, num_classes
    
    def _extract_gt_label(self, img_path):
        """
        이미지 경로에서 GT 라벨 추출
        경로에 'bad' 또는 'defect'가 있으면 불량(1), 'good'이 있으면 양품(0)
        """
        path_lower = img_path.lower()
        
        # 경로를 '/'로 분할하여 폴더명 확인
        parts = path_lower.split('/')
        
        if 'bad' in parts or 'defect' in parts:
            return 1  # 불량
        elif 'good' in parts:
            return 0  # 양품
        else:
            # 파일명에서도 확인
            filename = os.path.basename(path_lower)
            if 'bad' in filename or 'defect' in filename:
                return 1
            elif 'good' in filename:
                return 0
            else:
                return None  # GT를 알 수 없음
    
    def process_image_list(self, txt_file):
        """
        이미지 리스트 처리
        
        Args:
            txt_file (str): 이미지 경로가 담긴 txt 파일
        """
        # 결과 폴더 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = Path('runs') / f"{self.project_name}_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 하위 폴더 생성
        crops_dir = result_dir / 'crops'
        vis_dir = result_dir / 'visualizations'
        crops_dir.mkdir(exist_ok=True)
        vis_dir.mkdir(exist_ok=True)
        
        # 이미지 경로 읽기
        with open(txt_file, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        print(f"\n{'='*60}")
        print(f"🚀 이미지 처리 시작")
        print(f"{'='*60}")
        print(f"  - 총 이미지 수: {len(image_paths)}")
        print(f"  - 결과 저장 위치: {result_dir}\n")
        
        results = []
        y_true = []
        y_pred = []
        
        for idx, img_path in enumerate(tqdm(image_paths, desc="Processing")):
            if not os.path.exists(img_path):
                print(f"⚠️  이미지를 찾을 수 없습니다: {img_path}")
                continue
            
            # GT 라벨 추출
            gt_label = self._extract_gt_label(img_path)
            
            # 이미지 처리
            result = self.process_single_image(
                img_path, 
                result_dir, 
                crops_dir, 
                vis_dir, 
                idx,
                gt_label
            )
            results.append(result)
            
            # confusion matrix용 데이터 수집
            if gt_label is not None and result['status'] in ['processed', 'defect']:
                y_true.append(gt_label)
                pred_label = 1 if result['final_prediction'] == 'defect' else 0
                y_pred.append(pred_label)
                
                # 볼트 모드이고 4-class인 경우, 각 볼트의 클래스 예측도 저장
                if self.mode == 'bolt' and result.get('bolt_results'):
                    # 첫 번째 볼트 결과에서 num_classes 확인
                    first_bolt = result['bolt_results'][0] if result['bolt_results'] else None
                    if first_bolt and first_bolt.get('num_classes') == 4:
                        # 원본 클래스 예측 저장 (나중에 confusion matrix 생성용)
                        if not hasattr(self, 'bolt_y_true_class'):
                            self.bolt_y_true_class = []
                            self.bolt_y_pred_class = []
                        
                        # GT는 0/1만 있으므로, 4-class confusion matrix는 제한적
                        # GT가 0이면 클래스 0, GT가 1이면 첫 번째 볼트의 예측 클래스 사용
                        if gt_label == 0:
                            gt_class = 0  # 양품
                        else:
                            # 불량인 경우, 첫 번째 볼트의 예측 클래스 사용 (1,2,3 중 하나)
                            pred_class_val = first_bolt['pred_class']
                            gt_class = pred_class_val if pred_class_val != 0 else 1
                        
                        pred_class = first_bolt['pred_class']
                        self.bolt_y_true_class.append(gt_class)
                        self.bolt_y_pred_class.append(pred_class)
                
                # 도어 모드이고 5-class인 경우, 각 부위의 클래스 예측도 저장
                if self.mode == 'frontdoor' and result.get('parts'):
                    # 첫 번째 부위 결과에서 num_classes 확인
                    first_part_key = list(result['parts'].keys())[0] if result['parts'] else None
                    if first_part_key:
                        first_part = result['parts'][first_part_key]
                        if first_part and first_part.get('num_classes') == 5:
                            # 원본 클래스 예측 저장 (나중에 confusion matrix 생성용)
                            if not hasattr(self, 'door_y_true_class'):
                                self.door_y_true_class = []
                                self.door_y_pred_class = []
                            
                            # GT는 0/1만 있으므로, 5-class confusion matrix는 제한적
                            # GT가 0이면 클래스 0, GT가 1이면 첫 번째 부위의 예측 클래스 사용
                            if gt_label == 0:
                                gt_class = 0  # 양품
                            else:
                                # 불량인 경우, 첫 번째 부위의 예측 클래스 사용 (1,2,3,4 중 하나)
                                pred_class_val = first_part['pred_class']
                                gt_class = pred_class_val if pred_class_val != 0 else 1
                            
                            pred_class = first_part['pred_class']
                            self.door_y_true_class.append(gt_class)
                            self.door_y_pred_class.append(pred_class)
            
            # 메모리 정리 (매 10개 이미지마다)
            if (idx + 1) % 10 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 결과 저장
        self._save_results(results, result_dir)
        
        # Confusion Matrix 생성
        if len(y_true) > 0:
            self._plot_confusion_matrix(y_true, y_pred, result_dir)
            
            # 볼트 모드이고 4-class인 경우, 클래스별 confusion matrix도 생성
            if self.mode == 'bolt' and hasattr(self, 'bolt_y_true_class') and len(self.bolt_y_true_class) > 0:
                self._plot_bolt_class_confusion_matrix(self.bolt_y_true_class, self.bolt_y_pred_class, result_dir)
            
            # 도어 모드이고 5-class인 경우, 클래스별 confusion matrix도 생성
            if self.mode == 'frontdoor' and hasattr(self, 'door_y_true_class') and len(self.door_y_true_class) > 0:
                self._plot_door_class_confusion_matrix(self.door_y_true_class, self.door_y_pred_class, result_dir)
        
        # 통계 출력
        self._print_statistics(results, y_true, y_pred)
        
        return results
    
    def process_single_image(self, img_path, result_dir, crops_dir, vis_dir, idx, gt_label):
        """단일 이미지 처리"""
        try:
            # 이미지 로드
            img = cv2.imread(img_path)
            if img is None:
                return {
                    'image_path': img_path,
                    'status': 'error',
                    'message': 'Failed to load image',
                    'gt_label': gt_label
                }
            
            # YOLO 검출
            yolo_results = self.yolo_model.predict(
                img_path, 
                conf=self.conf_threshold,
                verbose=False
            )[0]
            
            if self.use_obb:
                # OBB 모드
                obbs = yolo_results.obb if hasattr(yolo_results, 'obb') else None
                if obbs is None:
                    return {
                        'image_path': img_path,
                        'status': 'error',
                        'message': 'OBB 모드인데 모델이 OBB를 지원하지 않습니다.',
                        'gt_label': gt_label
                    }
                
                # OBB 속성 확인 및 디버깅
                try:
                    if len(obbs) > 0:
                        first_obb = obbs[0]
                        # xywhr 속성이 있는지 확인
                        if not hasattr(first_obb, 'xywhr'):
                            return {
                                'image_path': img_path,
                                'status': 'error',
                                'message': f'OBB 객체에 xywhr 속성이 없습니다. 사용 가능한 속성: {dir(first_obb)}',
                                'gt_label': gt_label
                            }
                    
                    if self.mode == 'frontdoor':
                        result = self._process_frontdoor_obb(
                            img, img_path, obbs, crops_dir, vis_dir, idx, gt_label
                        )
                    elif self.mode == 'bolt':
                        result = self._process_bolt_obb(
                            img, img_path, obbs, crops_dir, vis_dir, idx, gt_label
                        )
                except Exception as e:
                    import traceback
                    return {
                        'image_path': img_path,
                        'status': 'error',
                        'message': f'OBB 처리 중 오류: {str(e)}',
                        'traceback': traceback.format_exc(),
                        'gt_label': gt_label
                    }
            else:
                # 일반 bbox 모드
                boxes = yolo_results.boxes
                
                if self.mode == 'frontdoor':
                    result = self._process_frontdoor(
                        img, img_path, boxes, crops_dir, vis_dir, idx, gt_label
                    )
                elif self.mode == 'bolt':
                    result = self._process_bolt(
                        img, img_path, boxes, crops_dir, vis_dir, idx, gt_label
                    )
            
            result['gt_label'] = gt_label
            return result
        
        except Exception as e:
            import traceback
            return {
                'image_path': img_path,
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc(),
                'gt_label': gt_label
            }
    
    def _process_frontdoor(self, img, img_path, boxes, crops_dir, vis_dir, idx, gt_label):
        """프론트도어 처리"""
        # 클래스별 검출 결과 정리
        detections = {'high': [], 'mid': [], 'low': []}
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            class_name = self.yolo_model.names[cls_id].lower()
            if class_name in detections:
                detections[class_name].append({
                    'bbox': xyxy,
                    'conf': conf
                })
        
        # 조건 확인: high/mid/low 각 1개씩 OR high/low 각 1개씩
        has_all_three = (len(detections['high']) == 1 and 
                        len(detections['mid']) == 1 and 
                        len(detections['low']) == 1)
        has_high_low = (len(detections['high']) == 1 and 
                       len(detections['low']) == 1 and 
                       len(detections['mid']) == 0)
        
        if not (has_all_three or has_high_low):
            # 시각화 (검출 실패)
            self._save_visualization(
                img, img_path, [], vis_dir, idx, 
                'skipped', gt_label, None, None
            )
            return {
                'image_path': img_path,
                'status': 'skipped',
                'message': 'Detection condition not met',
                'detections': {k: len(v) for k, v in detections.items()}
            }
        
        # 각 부위별 크롭 및 분류
        part_results = {}
        parts_to_process = ['high', 'mid', 'low'] if has_all_three else ['high', 'low']
        crop_info = []
        
        for part in parts_to_process:
            if len(detections[part]) > 0:
                bbox = detections[part][0]['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cropped = img[y1:y2, x1:x2]
                
                # 크롭 이미지 저장
                crop_filename = f"{idx:04d}_{part}.jpg"
                crop_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_path), cropped)
                
                # DINOv2 분류
                result = self._classify_with_dino(cropped, part)
                
                part_results[part] = {
                    'bbox': bbox.tolist(),
                    'yolo_conf': detections[part][0]['conf'],
                    'pred_class': result['pred_class'],
                    'confidence': result['confidence'],
                    'is_defect': result['is_defect'],
                    'defect_confidence': result['defect_confidence'],
                    'num_classes': result['num_classes'],
                    'crop_path': str(crop_path)
                }
                
                # 라벨 생성
                num_classes = result['num_classes']
                if num_classes == 5:
                    # 5-class 모드 (도어): good, shipping_seal, no_seal, work_seal, tape_seal
                    class_names = ['good', 'shipping_seal', 'no_seal', 'work_seal', 'tape_seal']
                    class_name = class_names[result['pred_class']] if result['pred_class'] < len(class_names) else str(result['pred_class'])
                    label = f"{part}: {class_name} ({result['confidence'][result['pred_class']]:.2f})"
                elif num_classes == 4:
                    # 4-class 모드 (볼트): frontside_good, frontside_bad, side_good, side_bad
                    class_names = ['frontside_good', 'frontside_bad', 'side_good', 'side_bad']
                    class_name = class_names[result['pred_class']] if result['pred_class'] < len(class_names) else str(result['pred_class'])
                    label = f"{part}: {class_name} ({result['confidence'][result['pred_class']]:.2f})"
                else:
                    # 2-class 모드
                    label = f"{part}: {'Bad' if result['is_defect'] else 'Good'} ({result['confidence'][result['pred_class']]:.2f})"
                
                crop_info.append({
                    'bbox': bbox,
                    'label': label,
                    'color': (0, 0, 255) if result['is_defect'] else (0, 255, 0)
                })
        
        # Voting
        if self.voting_method == 'hard':
            final_pred = self._hard_voting(part_results)
        else:  # soft
            final_pred = self._soft_voting(part_results)
        
        # 시각화 저장
        self._save_visualization(
            img, img_path, crop_info, vis_dir, idx, 
            final_pred, gt_label, part_results, None
        )
        
        return {
            'image_path': img_path,
            'status': 'processed',
            'mode': 'frontdoor',
            'parts': part_results,
            'final_prediction': final_pred,
            'voting_method': self.voting_method
        }
    
    def _process_bolt(self, img, img_path, boxes, crops_dir, vis_dir, idx, gt_label):
        """볼트 처리"""
        # 클래스별 검출 결과 정리
        bolt_detections = []  # 0, 1번 클래스 (볼트)
        frame_detections = []  # 2~7번 클래스 (프레임)
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            detection = {
                'class_id': cls_id,
                'class_name': self.bolt_class_names.get(cls_id, 'unknown'),
                'bbox': xyxy,
                'conf': conf,
                'center': [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
            }
            
            if cls_id in [0, 1]:  # 볼트
                bolt_detections.append(detection)
            elif cls_id in [2, 3, 4, 5, 6, 7]:  # 프레임
                frame_detections.append(detection)
        
        # 2~7번 프레임이 없으면 스킵
        if len(frame_detections) == 0:
            self._save_visualization(
                img, img_path, [], vis_dir, idx, 
                'skipped', gt_label, None, frame_detections
            )
            return {
                'image_path': img_path,
                'status': 'skipped',
                'message': 'No frame detection (class 2-7)',
                'bolt_count': len(bolt_detections),
                'frame_count': len(frame_detections)
            }
        
        # 각 프레임 영역 내의 볼트 찾기
        valid_bolts = []
        for frame in frame_detections:
            frame_bbox = frame['bbox']
            frame_cls = frame['class_id']
            
            # 이 프레임 내의 볼트들
            bolts_in_frame = []
            for bolt in bolt_detections:
                cx, cy = bolt['center']
                if (frame_bbox[0] <= cx <= frame_bbox[2] and 
                    frame_bbox[1] <= cy <= frame_bbox[3]):
                    bolts_in_frame.append(bolt)
            
            # 프레임 내의 모든 볼트를 양불량 판단에 사용
            valid_bolts.extend(bolts_in_frame)
        
        # 볼트가 없으면 불량
        if len(valid_bolts) == 0:
            self._save_visualization(
                img, img_path, [], vis_dir, idx, 
                'defect', gt_label, None, frame_detections
            )
            return {
                'image_path': img_path,
                'status': 'defect',
                'reason': 'no_bolts_in_frame',
                'final_prediction': 'defect'
            }
        
        # 각 볼트를 DINOv2로 분류
        bolt_results = []
        crop_info = []
        
        for bolt_idx, bolt in enumerate(valid_bolts):
            bbox = bolt['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cropped = img[y1:y2, x1:x2]
            
            # 크롭 이미지 저장
            crop_filename = f"{idx:04d}_bolt_{bolt_idx}.jpg"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), cropped)
            
            result = self._classify_with_dino(cropped, 'bolt')
            
            bolt_results.append({
                'bbox': bbox.tolist(),
                'yolo_class': bolt['class_name'],
                'yolo_conf': bolt['conf'],
                'pred_class': result['pred_class'],
                'confidence': result['confidence'],
                'is_defect': result['is_defect'],
                'defect_confidence': result['defect_confidence'],
                'num_classes': result['num_classes'],
                'crop_path': str(crop_path)
            })
            
            # 라벨 생성 (2-class 또는 4-class에 따라)
            num_classes = result['num_classes']
            if num_classes == 4:
                class_names = ['frontside_good', 'frontside_bad', 'side_good', 'side_bad']
                class_name = class_names[result['pred_class']] if result['pred_class'] < len(class_names) else str(result['pred_class'])
                label = f"Bolt: {class_name} ({result['confidence'][result['pred_class']]:.2f})"
            else:
                label = f"Bolt: {'Bad' if result['is_defect'] else 'Good'} ({result['confidence'][result['pred_class']]:.2f})"
            
            crop_info.append({
                'bbox': bbox,
                'label': label,
                'color': (0, 0, 255) if result['is_defect'] else (0, 255, 0)
            })
        
        # Voting 방식으로 최종 판정
        if self.voting_method == 'hard':
            final_pred = self._hard_voting_bolt(bolt_results)
        else:  # soft
            final_pred = self._soft_voting_bolt(bolt_results)
        
        # 시각화 저장
        self._save_visualization(
            img, img_path, crop_info, vis_dir, idx, 
            final_pred, gt_label, bolt_results, frame_detections
        )
        
        return {
            'image_path': img_path,
            'status': 'processed',
            'mode': 'bolt',
            'bolt_count': len(valid_bolts),
            'bolt_results': bolt_results,
            'final_prediction': final_pred
        }
    
    def _process_frontdoor_obb(self, img, img_path, obbs, crops_dir, vis_dir, idx, gt_label):
        """프론트도어 OBB 처리"""
        img_h, img_w = img.shape[:2]
        
        # 클래스별 검출 결과 정리
        detections = {'high': [], 'mid': [], 'low': []}
        
        # 빈 OBB 처리
        if len(obbs) == 0:
            self._save_visualization_obb(
                img, img_path, [], vis_dir, idx, 
                'skipped', gt_label, None, None
            )
            return {
                'image_path': img_path,
                'status': 'skipped',
                'message': 'No OBB detections',
                'detections': {k: 0 for k in detections.keys()}
            }
        
        for obb in obbs:
            try:
                cls_id = int(obb.cls[0])
                conf = float(obb.conf[0])
                # OBB 형식: xywhr (center_x, center_y, width, height, rotation)
                xywhr = obb.xywhr[0].cpu().numpy()
                if len(xywhr) < 5:
                    continue
                cx, cy, w, h, angle = xywhr[:5]
            
                # 정규화된 좌표를 절대 좌표로 변환 (xywhr는 정규화된 좌표라고 가정)
                # 만약 이미 절대 좌표라면 변환하지 않음 (w, h가 이미지 크기보다 크면 절대 좌표)
                if w > 1.0 or h > 1.0:
                    # 이미 절대 좌표
                    cx_abs, cy_abs, w_abs, h_abs = cx, cy, w, h
                else:
                    # 정규화된 좌표를 절대 좌표로 변환
                    cx_abs = cx * img_w
                    cy_abs = cy * img_h
                    w_abs = w * img_w
                    h_abs = h * img_h
                
                class_name = self.yolo_model.names[cls_id].lower()
                if class_name in detections:
                    detections[class_name].append({
                        'cx': cx_abs,
                        'cy': cy_abs,
                        'w': w_abs,
                        'h': h_abs,
                        'angle': angle,
                        'conf': conf
                    })
            except Exception as e:
                print(f"⚠️  OBB 처리 중 오류 (이미지: {os.path.basename(img_path)}): {e}")
                continue
        
        # 조건 확인: high/mid/low 각 1개씩 OR high/low 각 1개씩
        has_all_three = (len(detections['high']) == 1 and 
                        len(detections['mid']) == 1 and 
                        len(detections['low']) == 1)
        has_high_low = (len(detections['high']) == 1 and 
                       len(detections['low']) == 1 and 
                       len(detections['mid']) == 0)
        
        if not (has_all_three or has_high_low):
            # 시각화 (검출 실패)
            self._save_visualization_obb(
                img, img_path, [], vis_dir, idx, 
                'skipped', gt_label, None, None
            )
            return {
                'image_path': img_path,
                'status': 'skipped',
                'message': 'Detection condition not met',
                'detections': {k: len(v) for k, v in detections.items()}
            }
        
        # 각 부위별 크롭 및 분류
        part_results = {}
        parts_to_process = ['high', 'mid', 'low'] if has_all_three else ['high', 'low']
        crop_info = []
        
        for part in parts_to_process:
            if len(detections[part]) > 0:
                det = detections[part][0]
                cx, cy, w, h, angle = det['cx'], det['cy'], det['w'], det['h'], det['angle']
                
                # 회전된 객체 crop (도어 모드: part 정보 전달)
                cropped = crop_rotated_object(img, cx, cy, w, h, angle, part=part)
                if cropped is None:
                    continue
                
                # 크롭 이미지 저장
                crop_filename = f"{idx:04d}_{part}.jpg"
                crop_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_path), cropped)
                
                # DINOv2 분류
                result = self._classify_with_dino(cropped, part)
                
                part_results[part] = {
                    'cx': float(cx),
                    'cy': float(cy),
                    'w': float(w),
                    'h': float(h),
                    'angle': float(angle),
                    'yolo_conf': det['conf'],
                    'pred_class': result['pred_class'],
                    'confidence': result['confidence'],
                    'is_defect': result['is_defect'],
                    'defect_confidence': result['defect_confidence'],
                    'num_classes': result['num_classes'],
                    'crop_path': str(crop_path)
                }
                
                # 라벨 생성
                num_classes = result['num_classes']
                if num_classes == 5:
                    # 5-class 모드 (도어): good, shipping_seal, no_seal, work_seal, tape_seal
                    class_names = ['good', 'shipping_seal', 'no_seal', 'work_seal', 'tape_seal']
                    class_name = class_names[result['pred_class']] if result['pred_class'] < len(class_names) else str(result['pred_class'])
                    label = f"{part}: {class_name} ({result['confidence'][result['pred_class']]:.2f})"
                elif num_classes == 4:
                    # 4-class 모드 (볼트): frontside_good, frontside_bad, side_good, side_bad
                    class_names = ['frontside_good', 'frontside_bad', 'side_good', 'side_bad']
                    class_name = class_names[result['pred_class']] if result['pred_class'] < len(class_names) else str(result['pred_class'])
                    label = f"{part}: {class_name} ({result['confidence'][result['pred_class']]:.2f})"
                else:
                    # 2-class 모드
                    label = f"{part}: {'Bad' if result['is_defect'] else 'Good'} ({result['confidence'][result['pred_class']]:.2f})"
                
                crop_info.append({
                    'cx': cx,
                    'cy': cy,
                    'w': w,
                    'h': h,
                    'angle': angle,
                    'label': label,
                    'color': (0, 0, 255) if result['is_defect'] else (0, 255, 0)
                })
        
        # Voting
        if self.voting_method == 'hard':
            final_pred = self._hard_voting(part_results)
        else:  # soft
            final_pred = self._soft_voting(part_results)
        
        # 시각화 저장
        self._save_visualization_obb(
            img, img_path, crop_info, vis_dir, idx, 
            final_pred, gt_label, part_results, None
        )
        
        return {
            'image_path': img_path,
            'status': 'processed',
            'mode': 'frontdoor',
            'obb_mode': True,
            'parts': part_results,
            'final_prediction': final_pred,
            'voting_method': self.voting_method
        }
    
    def _process_bolt_obb(self, img, img_path, obbs, crops_dir, vis_dir, idx, gt_label):
        """볼트 OBB 처리"""
        img_h, img_w = img.shape[:2]
        
        # 클래스별 검출 결과 정리
        bolt_detections = []  # 0, 1번 클래스 (볼트)
        frame_detections = []  # 2~7번 클래스 (프레임)
        
        # 빈 OBB 처리
        if len(obbs) == 0:
            self._save_visualization_obb(
                img, img_path, [], vis_dir, idx, 
                'skipped', gt_label, None
            )
            return {
                'image_path': img_path,
                'status': 'skipped',
                'message': 'No OBB detections',
                'bolt_count': 0,
                'frame_count': 0
            }
        
        for obb in obbs:
            try:
                cls_id = int(obb.cls[0])
                conf = float(obb.conf[0])
                xywhr = obb.xywhr[0].cpu().numpy()
                if len(xywhr) < 5:
                    continue
                cx, cy, w, h, angle = xywhr[:5]
                
                # 정규화된 좌표를 절대 좌표로 변환 (xywhr는 정규화된 좌표라고 가정)
                # 만약 이미 절대 좌표라면 변환하지 않음 (w, h가 이미지 크기보다 크면 절대 좌표)
                if w > 1.0 or h > 1.0:
                    # 이미 절대 좌표
                    cx_abs, cy_abs, w_abs, h_abs = cx, cy, w, h
                else:
                    # 정규화된 좌표를 절대 좌표로 변환
                    cx_abs = cx * img_w
                    cy_abs = cy * img_h
                    w_abs = w * img_w
                    h_abs = h * img_h
            
                detection = {
                    'class_id': cls_id,
                    'class_name': self.bolt_class_names.get(cls_id, 'unknown'),
                    'cx': cx_abs,
                    'cy': cy_abs,
                    'w': w_abs,
                    'h': h_abs,
                    'angle': angle,
                    'conf': conf,
                    'center': [cx_abs, cy_abs]
                }
                
                if cls_id in [0, 1]:  # 볼트
                    bolt_detections.append(detection)
                elif cls_id in [2, 3, 4, 5, 6, 7]:  # 프레임
                    frame_detections.append(detection)
            except Exception as e:
                print(f"⚠️  OBB 처리 중 오류 (이미지: {os.path.basename(img_path)}): {e}")
                continue
        
        # 2~7번 프레임이 없으면 스킵
        if len(frame_detections) == 0:
            self._save_visualization_obb(
                img, img_path, [], vis_dir, idx, 
                'skipped', gt_label, None, frame_detections
            )
            return {
                'image_path': img_path,
                'status': 'skipped',
                'message': 'No frame detection (class 2-7)',
                'bolt_count': len(bolt_detections),
                'frame_count': len(frame_detections)
            }
        
        # 각 프레임 영역 내의 볼트 찾기
        valid_bolts = []
        for frame in frame_detections:
            frame_cx, frame_cy = frame['cx'], frame['cy']
            frame_w, frame_h = frame['w'], frame['h']
            frame_angle = frame['angle']
            frame_cls = frame['class_id']
            
            # 이 프레임 내의 볼트들 (회전된 프레임 내부 확인)
            bolts_in_frame = []
            for bolt in bolt_detections:
                bolt_cx, bolt_cy = bolt['center']
                # 회전된 프레임 내부에 볼트가 있는지 확인
                if point_in_rotated_box(bolt_cx, bolt_cy, frame_cx, frame_cy, 
                                        frame_w, frame_h, frame_angle):
                    bolts_in_frame.append(bolt)
            
            # 프레임 내의 모든 볼트를 양불량 판단에 사용
            valid_bolts.extend(bolts_in_frame)
        
        # 볼트가 없으면 불량
        if len(valid_bolts) == 0:
            self._save_visualization_obb(
                img, img_path, [], vis_dir, idx, 
                'defect', gt_label, None, frame_detections
            )
            return {
                'image_path': img_path,
                'status': 'defect',
                'reason': 'no_bolts_in_frame',
                'final_prediction': 'defect'
            }
        
        # 각 볼트를 DINOv2로 분류
        bolt_results = []
        crop_info = []
        
        for bolt_idx, bolt in enumerate(valid_bolts):
            cx, cy, w, h, angle = bolt['cx'], bolt['cy'], bolt['w'], bolt['h'], bolt['angle']
            
            # 회전된 객체 crop
            cropped = crop_rotated_object(img, cx, cy, w, h, angle)
            if cropped is None:
                continue
            
            # 크롭 이미지 저장
            crop_filename = f"{idx:04d}_bolt_{bolt_idx}.jpg"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), cropped)
            
            result = self._classify_with_dino(cropped, 'bolt')
            
            bolt_results.append({
                'cx': float(cx),
                'cy': float(cy),
                'w': float(w),
                'h': float(h),
                'angle': float(angle),
                'yolo_class': bolt['class_name'],
                'yolo_conf': bolt['conf'],
                'pred_class': result['pred_class'],
                'confidence': result['confidence'],
                'is_defect': result['is_defect'],
                'defect_confidence': result['defect_confidence'],
                'num_classes': result['num_classes'],
                'crop_path': str(crop_path)
            })
            
            # 라벨 생성 (2-class 또는 4-class에 따라)
            num_classes = result['num_classes']
            if num_classes == 4:
                class_names = ['frontside_good', 'frontside_bad', 'side_good', 'side_bad']
                class_name = class_names[result['pred_class']] if result['pred_class'] < len(class_names) else str(result['pred_class'])
                label = f"Bolt: {class_name} ({result['confidence'][result['pred_class']]:.2f})"
            else:
                label = f"Bolt: {'Bad' if result['is_defect'] else 'Good'} ({result['confidence'][result['pred_class']]:.2f})"
            
            crop_info.append({
                'cx': cx,
                'cy': cy,
                'w': w,
                'h': h,
                'angle': angle,
                'label': label,
                'color': (0, 0, 255) if result['is_defect'] else (0, 255, 0)
            })
        
        # Voting 방식으로 최종 판정
        if self.voting_method == 'hard':
            final_pred = self._hard_voting_bolt(bolt_results)
        else:  # soft
            final_pred = self._soft_voting_bolt(bolt_results)
        
        # 시각화 저장
        self._save_visualization_obb(
            img, img_path, crop_info, vis_dir, idx, 
            final_pred, gt_label, bolt_results, frame_detections
        )
        
        return {
            'image_path': img_path,
            'status': 'processed',
            'mode': 'bolt',
            'obb_mode': True,
            'bolt_count': len(valid_bolts),
            'bolt_results': bolt_results,
            'final_prediction': final_pred
        }
    
    def _save_visualization(self, img, img_path, crop_info, vis_dir, idx, 
                           prediction, gt_label, detail_results, frame_detections=None):
        """시각화 이미지 저장"""
        vis_img = img.copy()
        
        # 프레임 바운딩 박스 그리기
        if frame_detections:
            for frame in frame_detections:
                bbox = frame['bbox']
                class_name = frame['class_name']
                conf = frame['conf']
                x1, y1, x2, y2 = map(int, bbox)
                # 프레임은 파란색으로 표시
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # 프레임 라벨
                frame_label = f"{class_name} ({conf:.2f})"
                label_size, _ = cv2.getTextSize(frame_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (255, 0, 0), -1)
                cv2.putText(vis_img, frame_label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 볼트 바운딩 박스 그리기
        for crop in crop_info:
            bbox = crop['bbox']
            label = crop['label']
            color = crop['color']
            
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 배경
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 오른쪽 상단에 결과 표시
        h, w = vis_img.shape[:2]
        
        # GT vs Prediction 비교
        if gt_label is not None:
            gt_text = "GT: Good" if gt_label == 0 else "GT: Bad"
            pred_text = f"Pred: {prediction.capitalize()}"
            
            # 정답 여부 판단
            pred_label = 1 if prediction == 'defect' else 0
            is_correct = (gt_label == pred_label)
            result_symbol = "✓" if is_correct else "✗"
            result_color = (0, 255, 0) if is_correct else (0, 0, 255)
            
            # 배경 사각형
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 110), (0, 0, 0), -1)
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 110), (255, 255, 255), 2)
            
            # 텍스트
            cv2.putText(vis_img, gt_text, (w - 240, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_img, pred_text, (w - 240, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_img, result_symbol, (w - 240, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 3)
        else:
            # GT 없는 경우
            pred_text = f"Pred: {prediction.capitalize()}"
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 60), (0, 0, 0), -1)
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 60), (255, 255, 255), 2)
            cv2.putText(vis_img, pred_text, (w - 240, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 저장
        vis_filename = f"{idx:04d}_vis.jpg"
        vis_path = vis_dir / vis_filename
        cv2.imwrite(str(vis_path), vis_img)
    
    def _save_visualization_obb(self, img, img_path, crop_info, vis_dir, idx, 
                                prediction, gt_label, detail_results, frame_detections=None):
        """OBB 시각화 이미지 저장"""
        vis_img = img.copy()
        
        # 프레임 회전된 바운딩 박스 그리기
        if frame_detections:
            for frame in frame_detections:
                cx = frame['cx']
                cy = frame['cy']
                w = frame['w']
                h = frame['h']
                angle = frame['angle']
                class_name = frame['class_name']
                conf = frame['conf']
                
                # 회전된 프레임 박스의 모서리 계산
                corners = compute_rotated_box_corners(cx, cy, w, h, angle)
                corners_int = np.array(corners, dtype=np.int32)
                
                # 다각형 그리기 (프레임은 파란색)
                cv2.polylines(vis_img, [corners_int], isClosed=True, color=(255, 0, 0), thickness=2)
                
                # 프레임 라벨
                frame_label = f"{class_name} ({conf:.2f})"
                x1, y1 = int(corners[0][0]), int(corners[0][1])
                label_size, _ = cv2.getTextSize(frame_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (255, 0, 0), -1)
                cv2.putText(vis_img, frame_label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 볼트 회전된 바운딩 박스 그리기
        for crop in crop_info:
            cx = crop['cx']
            cy = crop['cy']
            w = crop['w']
            h = crop['h']
            angle = crop['angle']
            label = crop['label']
            color = crop['color']
            
            # 회전된 박스의 모서리 계산
            corners = compute_rotated_box_corners(cx, cy, w, h, angle)
            corners_int = np.array(corners, dtype=np.int32)
            
            # 다각형 그리기
            cv2.polylines(vis_img, [corners_int], isClosed=True, color=color, thickness=2)
            
            # 라벨 배경 (첫 번째 모서리 위에)
            x1, y1 = int(corners[0][0]), int(corners[0][1])
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 오른쪽 상단에 결과 표시
        h, w = vis_img.shape[:2]
        
        # GT vs Prediction 비교
        if gt_label is not None:
            gt_text = "GT: Good" if gt_label == 0 else "GT: Bad"
            pred_text = f"Pred: {prediction.capitalize()}"
            
            # 정답 여부 판단
            pred_label = 1 if prediction == 'defect' else 0
            is_correct = (gt_label == pred_label)
            result_symbol = "✓" if is_correct else "✗"
            result_color = (0, 255, 0) if is_correct else (0, 0, 255)
            
            # 배경 사각형
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 110), (0, 0, 0), -1)
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 110), (255, 255, 255), 2)
            
            # 텍스트
            cv2.putText(vis_img, gt_text, (w - 240, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_img, pred_text, (w - 240, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_img, result_symbol, (w - 240, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 3)
        else:
            # GT 없는 경우
            pred_text = f"Pred: {prediction.capitalize()}"
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 60), (0, 0, 0), -1)
            cv2.rectangle(vis_img, (w - 250, 10), (w - 10, 60), (255, 255, 255), 2)
            cv2.putText(vis_img, pred_text, (w - 240, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 저장
        vis_filename = f"{idx:04d}_vis.jpg"
        vis_path = vis_dir / vis_filename
        cv2.imwrite(str(vis_path), vis_img)
    
    def _classify_with_dino(self, cropped_img, part):
        """DINOv2로 크롭된 이미지 분류"""
        num_classes = self.dino_num_classes.get(part, 2)
        
        if cropped_img.size == 0:
            # 빈 이미지는 불량으로
            if num_classes == 5:
                confidence = [0.0, 0.0, 0.0, 0.0, 1.0]  # 클래스 4에 높은 confidence
                pred_class = 4
                is_defect = True
                defect_confidence = 1.0
            elif num_classes == 4:
                confidence = [0.0, 0.0, 0.0, 1.0]  # 클래스 3에 높은 confidence
                pred_class = 3
                is_defect = True
                defect_confidence = 1.0
            else:
                confidence = [0.0, 1.0]
                pred_class = 1
                is_defect = True
                defect_confidence = 1.0
            return {
                'pred_class': pred_class,
                'confidence': confidence,
                'is_defect': is_defect,
                'defect_confidence': defect_confidence,
                'num_classes': num_classes
            }
        
        # BGR to RGB
        cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)
        
        # 전처리
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.dino_models[part](img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0].cpu().numpy().tolist()
        
        # 양불량 판정
        if num_classes == 5:
            # 5-class 모드 (도어): 0=양품, 1,2,3,4=불량
            is_defect = (pred_class != 0)
            # 불량 클래스들의 confidence 합계 계산 (소프트 보팅용)
            defect_confidence = sum(confidence[1:5]) if len(confidence) >= 5 else sum(confidence[1:]) if len(confidence) > 1 else 0.0
        elif num_classes == 4:
            # 4-class 모드 (볼트): 0=양품, 1,2,3=불량
            is_defect = (pred_class != 0)
            # 불량 클래스들의 confidence 합계 계산 (소프트 보팅용)
            defect_confidence = sum(confidence[1:4]) if len(confidence) >= 4 else confidence[1] if len(confidence) >= 2 else 0.0
        else:
            # 2-class 모드: 0=양품, 1=불량
            is_defect = (pred_class == 1)
            defect_confidence = confidence[1] if len(confidence) >= 2 else 0.0
        
        return {
            'pred_class': pred_class,
            'confidence': confidence,
            'is_defect': is_defect,
            'defect_confidence': defect_confidence,
            'num_classes': num_classes
        }
    
    def _hard_voting(self, part_results):
        """Hard Voting: 하나라도 불량이면 불량"""
        # 각 부위의 is_defect 확인
        has_defect = any(result.get('is_defect', result['pred_class'] != 0) for result in part_results.values())
        return 'defect' if has_defect else 'good'
    
    def _soft_voting(self, part_results):
        """Soft Voting: 평균 confidence"""
        # 각 부위의 불량 confidence 평균
        defect_confidences = [result.get('defect_confidence', result['confidence'][1] if len(result['confidence']) >= 2 else 0.0) 
                             for result in part_results.values()]
        avg_defect_conf = sum(defect_confidences) / len(defect_confidences) if len(defect_confidences) > 0 else 0.0
        
        # 평균이 0.5 이상이면 불량
        if avg_defect_conf >= 0.5:
            return 'defect'
        else:
            return 'good'
    
    def _hard_voting_bolt(self, bolt_results):
        """Hard Voting for Bolt: 하나라도 불량이면 불량"""
        if len(bolt_results) == 0:
            return 'good'
        
        has_defect = any(b.get('is_defect', b['pred_class'] == 1) for b in bolt_results)
        return 'defect' if has_defect else 'good'
    
    def _soft_voting_bolt(self, bolt_results):
        """Soft Voting for Bolt: 평균 불량 confidence"""
        if len(bolt_results) == 0:
            return 'good'
        
        # 각 볼트의 불량 confidence 평균
        defect_confidences = [b.get('defect_confidence', b['confidence'][1] if len(b['confidence']) >= 2 else 0.0) 
                             for b in bolt_results]
        avg_defect_conf = sum(defect_confidences) / len(defect_confidences) if len(defect_confidences) > 0 else 0.0
        
        # 평균이 0.5 이상이면 불량
        if avg_defect_conf >= 0.5:
            return 'defect'
        else:
            return 'good'
    
    def _save_results(self, results, result_dir):
        """결과 저장"""
        output_file = result_dir / 'results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 결과 저장: {output_file}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, result_dir):
        """Confusion Matrix 생성 및 저장 (양품/불량 2-class)"""
        # Confusion Matrix 계산
        cm = [[0, 0], [0, 0]]  # [[TN, FP], [FN, TP]]
        
        for true, pred in zip(y_true, y_pred):
            cm[true][pred] += 1
        
        # 시각화
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Good', 'Defect'],
                   yticklabels=['Good', 'Defect'],
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Good/Defect)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        cm_path = result_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion Matrix 저장: {cm_path}")
        
        # 정규화된 Confusion Matrix 생성
        cm_normalized = [[0.0, 0.0], [0.0, 0.0]]
        total_samples = len(y_true)
        if total_samples > 0:
            for i in range(2):
                row_sum = sum(cm[i])
                if row_sum > 0:
                    for j in range(2):
                        cm_normalized[i][j] = cm[i][j] / row_sum
        
        # 정규화된 버전 시각화
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                   xticklabels=['Good', 'Defect'],
                   yticklabels=['Good', 'Defect'],
                   ax=ax, cbar_kws={'label': 'Normalized Count'}, vmin=0, vmax=1)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Normalized Confusion Matrix (Good/Defect)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        cm_norm_path = result_dir / 'confusion_matrix_normalized.png'
        plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Normalized Confusion Matrix 저장: {cm_norm_path}")
        
        # 메트릭 계산
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 메트릭 저장
        metrics = {
            'confusion_matrix': {
                'TN': int(tn), 'FP': int(fp),
                'FN': int(fn), 'TP': int(tp)
            },
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        }
        
        metrics_path = result_dir / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ 메트릭 저장: {metrics_path}")
        
        return metrics
    
    def _plot_bolt_class_confusion_matrix(self, y_true_class, y_pred_class, result_dir):
        """볼트 4-class 모드용 클래스별 Confusion Matrix 생성 및 저장"""
        num_classes = 4
        class_names = ['frontside_good', 'frontside_bad', 'side_good', 'side_bad']
        
        # Confusion Matrix 계산
        cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for true, pred in zip(y_true_class, y_pred_class):
            if 0 <= true < num_classes and 0 <= pred < num_classes:
                cm[true][pred] += 1
        
        # 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cm_np = np.array(cm, dtype=np.int32)
        
        # Heatmap
        sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Bolt 4-Class)', fontsize=14, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        cm_path = result_dir / 'confusion_matrix_bolt_4class.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Bolt 4-Class Confusion Matrix 저장: {cm_path}")
        
        # 정규화된 Confusion Matrix 생성
        cm_normalized = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
        for i in range(num_classes):
            row_sum = sum(cm[i])
            if row_sum > 0:
                for j in range(num_classes):
                    cm_normalized[i][j] = cm[i][j] / row_sum
        
        # 정규화된 버전 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        cm_norm_np = np.array(cm_normalized, dtype=np.float32)
        
        sns.heatmap(cm_norm_np, annot=True, fmt='.3f', cmap='Blues', 
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Normalized Count'}, vmin=0, vmax=1)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Normalized Confusion Matrix (Bolt 4-Class)', fontsize=14, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        cm_norm_path = result_dir / 'confusion_matrix_bolt_4class_normalized.png'
        plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Bolt 4-Class Normalized Confusion Matrix 저장: {cm_norm_path}")
        
        # 메트릭 계산 (각 클래스별)
        class_metrics = {}
        for i in range(num_classes):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(num_classes) if j != i)
            fn = sum(cm[i][j] for j in range(num_classes) if j != i)
            tn = sum(cm[j][k] for j in range(num_classes) for k in range(num_classes) if j != i and k != i)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_names[i]] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        
        # 전체 정확도
        total_correct = sum(cm[i][i] for i in range(num_classes))
        total_samples = len(y_true_class)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # 메트릭 저장
        metrics = {
            'confusion_matrix_4class': cm,
            'class_metrics': class_metrics,
            'overall_accuracy': float(overall_accuracy)
        }
        
        metrics_path = result_dir / 'metrics_bolt_4class.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Bolt 4-Class 메트릭 저장: {metrics_path}")
        
        return metrics
    
    def _plot_door_class_confusion_matrix(self, y_true_class, y_pred_class, result_dir):
        """도어 5-class 모드용 클래스별 Confusion Matrix 생성 및 저장"""
        num_classes = 5
        class_names = ['good', 'shipping_seal', 'no_seal', 'work_seal', 'tape_seal']
        
        # Confusion Matrix 계산
        cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for true, pred in zip(y_true_class, y_pred_class):
            if 0 <= true < num_classes and 0 <= pred < num_classes:
                cm[true][pred] += 1
        
        # 시각화
        fig, ax = plt.subplots(figsize=(12, 10))
        
        cm_np = np.array(cm, dtype=np.int32)
        
        # Heatmap
        sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix (Door 5-Class)', fontsize=14, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        cm_path = result_dir / 'confusion_matrix_door_5class.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Door 5-Class Confusion Matrix 저장: {cm_path}")
        
        # 정규화된 Confusion Matrix 생성
        cm_normalized = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
        for i in range(num_classes):
            row_sum = sum(cm[i])
            if row_sum > 0:
                for j in range(num_classes):
                    cm_normalized[i][j] = cm[i][j] / row_sum
        
        # 정규화된 버전 시각화
        fig, ax = plt.subplots(figsize=(12, 10))
        cm_norm_np = np.array(cm_normalized, dtype=np.float32)
        
        sns.heatmap(cm_norm_np, annot=True, fmt='.3f', cmap='Blues', 
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Normalized Count'}, vmin=0, vmax=1)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Normalized Confusion Matrix (Door 5-Class)', fontsize=14, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        cm_norm_path = result_dir / 'confusion_matrix_door_5class_normalized.png'
        plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Door 5-Class Normalized Confusion Matrix 저장: {cm_norm_path}")
        
        # 메트릭 계산 (각 클래스별)
        class_metrics = {}
        for i in range(num_classes):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(num_classes) if j != i)
            fn = sum(cm[i][j] for j in range(num_classes) if j != i)
            tn = sum(cm[j][k] for j in range(num_classes) for k in range(num_classes) if j != i and k != i)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_names[i]] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        
        # 전체 정확도
        total_correct = sum(cm[i][i] for i in range(num_classes))
        total_samples = len(y_true_class)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # 메트릭 저장
        metrics = {
            'confusion_matrix_5class': cm,
            'class_metrics': class_metrics,
            'overall_accuracy': float(overall_accuracy)
        }
        
        metrics_path = result_dir / 'metrics_door_5class.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Door 5-Class 메트릭 저장: {metrics_path}")
        
        return metrics
    
    def _print_statistics(self, results, y_true, y_pred):
        """통계 출력"""
        print(f"\n{'='*60}")
        print("📊 처리 통계")
        print(f"{'='*60}")
        
        # 상태별 통계
        status_counts = {}
        for result in results:
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\n[처리 상태별 통계]")
        for status, count in sorted(status_counts.items()):
            print(f"  - {status}: {count}개")
        
        # 성능 메트릭
        if len(y_true) > 0 and len(y_pred) > 0:
            cm = [[0, 0], [0, 0]]  # [[TN, FP], [FN, TP]]
            for true, pred in zip(y_true, y_pred):
                cm[true][pred] += 1
            
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            total = tn + fp + fn + tp
            
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n[성능 메트릭]")
            print(f"  - Accuracy:  {accuracy:.4f} ({tp + tn}/{total})")
            print(f"  - Precision: {precision:.4f} ({tp}/{tp + fp})")
            print(f"  - Recall:    {recall:.4f} ({tp}/{tp + fn})")
            print(f"  - F1 Score:  {f1:.4f}")
            print(f"\n[Confusion Matrix]")
            print(f"  True Negative (TN):  {tn}")
            print(f"  False Positive (FP): {fp}")
            print(f"  False Negative (FN): {fn}")
            print(f"  True Positive (TP):  {tp}")
        else:
            print("\n[성능 메트릭]")
            print("  GT 라벨이 없어 메트릭을 계산할 수 없습니다.")
        
        print(f"\n{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + DINOv2 테스트 파이프라인")
    parser.add_argument("--config", required=True, type=str, help="모델 경로들이 들어있는 YAML 파일 경로")
    parser.add_argument("--txt", required=True, type=str, help="처리할 이미지 경로 목록이 담긴 txt 파일 경로")
    parser.add_argument("--mode", required=True, choices=["frontdoor", "door", "bolt"], help="실행 모드 (door는 frontdoor의 별칭)")
    parser.add_argument("--voting", default="hard", choices=["soft", "hard"], help="보팅 방식 (기본값: hard)")
    parser.add_argument("--project", default="pipeline_test", type=str, help="runs 하위 결과 폴더명 prefix")
    parser.add_argument("--conf", default=0.25, type=float, help="YOLO 신뢰도 임계값")
    parser.add_argument("--device", default="cuda", type=str, help="디바이스 (cuda|cpu)")
    parser.add_argument("--obb", action="store_true", help="OBB(Oriented Bounding Box) 모드 사용")
    return parser.parse_args()


def load_models_from_yaml(config_path, mode):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # YAML은 모델 경로만 포함한다고 가정
    yolo_model_path = cfg.get("yolo_model") or cfg.get("yolo") or cfg.get("yolo_model_path")
    if yolo_model_path is None:
        raise ValueError("YAML에 'yolo_model' 경로가 필요합니다.")

    # 'door'를 'frontdoor'로 정규화
    if mode.lower() == "door":
        mode = "frontdoor"

    dino_models = {}
    if mode == "frontdoor":
        # 예상 키: high, mid, low
        for key in ["high", "mid", "low"]:
            if key not in cfg:
                raise ValueError("frontdoor 모드는 YAML에 'high', 'mid', 'low' 키가 필요합니다.")
            dino_models[key] = cfg[key]
    else:
        # bolt 모드: bolt 단일 키
        bolt_path = cfg.get("bolt")
        if bolt_path is None:
            raise ValueError("bolt 모드는 YAML에 'bolt' 키가 필요합니다.")
        dino_models["bolt"] = bolt_path

    return yolo_model_path, dino_models


def main():
    args = parse_args()

    # YAML에서 경로들 로드
    yolo_model_path, dino_models = load_models_from_yaml(args.config, args.mode)

    # 파이프라인 생성
    pipeline = YOLODINOPipeline(
        mode=args.mode,
        yolo_model_path=yolo_model_path,
        dino_models=dino_models,
        device=args.device,
        conf_threshold=args.conf,
        voting_method=args.voting,
        project_name=args.project,
        use_obb=args.obb,
    )

    # 이미지 리스트 처리
    pipeline.process_image_list(args.txt)


if __name__ == "__main__":
    main()