import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import time
from datetime import datetime
import yaml
import math
from math import cos, sin


class DINOv2Classifier(nn.Module):
    """DINOv2 분류 모델"""
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


class RealtimeInspectionSystem:
    def __init__(self, mode='frontdoor', yolo_model_path=None, dino_models=None,
                 device='cuda', conf_threshold=0.25, voting_method='soft', use_obb=False):
        """
        실시간 카메라 검사 시스템
        
        Args:
            mode (str): 'frontdoor' 또는 'bolt'
            yolo_model_path (str): YOLO 모델 경로
            dino_models (dict): DINOv2 모델 경로들
            device (str): 디바이스
            conf_threshold (float): YOLO 신뢰도 임계값
            voting_method (str): 'hard' 또는 'soft'
            use_obb (bool): OBB(Oriented Bounding Box) 모드 사용 여부
        """
        self.mode = mode.lower()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.voting_method = voting_method
        self.use_obb = use_obb
        
        # YOLO 모델 로드
        print(f"🔄 YOLO 모델 로드 중: {yolo_model_path}")
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print(f"✓ YOLO 모델 로드 완료")
            if hasattr(self.yolo_model, 'names'):
                print(f"  - 클래스 수: {len(self.yolo_model.names)}")
                print(f"  - 클래스 목록: {list(self.yolo_model.names.values())}")
        except Exception as e:
            print(f"❌ YOLO 모델 로드 실패: {e}")
            raise
        
        # DINOv2 모델 로드 및 클래스 수 확인
        self.dino_models = {}
        self.dino_num_classes = {}  # 각 모델의 클래스 수 저장
        
        if self.mode == 'frontdoor':
            for part in ['high', 'mid', 'low']:
                print(f"🔄 DINOv2 모델 로드 중 ({part}): {dino_models[part]}")
                model, num_classes = self._load_dino_model(dino_models[part])
                self.dino_models[part] = model
                self.dino_num_classes[part] = num_classes
        else:  # bolt
            print(f"🔄 DINOv2 모델 로드 중 (bolt): {dino_models['bolt']}")
            model, num_classes = self._load_dino_model(dino_models['bolt'])
            self.dino_models['bolt'] = model
            self.dino_num_classes['bolt'] = num_classes
        
        # DINOv2 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 조건 체크 변수
        self.condition_start_time = None
        self.condition_met = False
        self.last_valid_frame = None
        self.last_valid_detections = None
        
        # 타임아웃 설정
        if self.mode == 'frontdoor':
            self.required_duration = 3.0  # 3초
        else:  # bolt
            self.required_duration = 5.0  # 5초
        
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
        
        # DINO 모드 확인 (config에서 읽어온 값 사용)
        self.dino_mode = None  # 나중에 config에서 설정
        
        print(f"✓ 실시간 검사 시스템 초기화 완료")
        print(f"  - 모드: {self.mode}")
        print(f"  - 디바이스: {self.device}")
        print(f"  - YOLO 신뢰도: {self.conf_threshold}")
        print(f"  - 조건 유지 시간: {self.required_duration}초")
        print(f"  - Voting 방법: {self.voting_method}")
        if self.use_obb:
            print(f"  - OBB 모드: 활성화")
        
        # DINO 클래스 수 출력
        if self.mode == 'frontdoor':
            for part in ['high', 'mid', 'low']:
                num_cls = self.dino_num_classes.get(part, 2)
                mode_text = "4-class" if num_cls == 4 else "2-class (simple)"
                print(f"  - DINO {part}: {mode_text}")
        else:
            # 볼트는 항상 2-class
            print(f"  - DINO bolt: 2-class (simple)")
    
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
    
    def run(self, source=0):
        """
        실시간 검사 실행
        
        Args:
            source: 카메라 소스 (0: 웹캠, 또는 RTSP URL 등)
        """
        print(f"\n{'='*60}")
        print(f"🎥 카메라 시작: {source}")
        print(f"{'='*60}\n")
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"❌ 카메라를 열 수 없습니다: {source}")
            return
        
        print(f"✓ 카메라 연결 성공")
        print(f"📋 대기 중... (조건이 만족되면 자동으로 캡처됩니다)")
        print(f"   종료하려면 'q' 키를 누르세요\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  프레임을 읽을 수 없습니다")
                    break
                
                # BGR to RGB 변환 (OpenCV는 BGR, YOLO는 RGB 기대)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # YOLO 검출
                if self.use_obb:
                    results = self.yolo_model.predict(
                        frame_rgb, 
                        conf=self.conf_threshold,
                        verbose=False,
                        task='obb'
                    )[0]
                else:
                    results = self.yolo_model.predict(
                        frame_rgb, 
                        conf=self.conf_threshold,
                        verbose=False
                    )[0]
                
                # 검출 결과 확인
                boxes = None
                if hasattr(results, 'boxes'):
                    boxes = results.boxes
                elif self.use_obb and hasattr(results, 'obb'):
                    boxes = results.obb
                
                # 조건 확인
                condition_satisfied, detections = self._check_condition(boxes)
                
                # 화면에 표시
                display_frame = self._draw_detections(frame.copy(), boxes)
                
                # 조건 만족 여부에 따른 처리
                if condition_satisfied:
                    if not self.condition_met:
                        # 조건이 처음 만족됨
                        self.condition_met = True
                        self.condition_start_time = time.time()
                        print(f"✓ 조건 만족! 타이머 시작...")
                    
                    # 경과 시간 계산
                    elapsed = time.time() - self.condition_start_time
                    
                    # 유효한 프레임 저장
                    self.last_valid_frame = frame.copy()
                    self.last_valid_detections = detections
                    
                    # 타이머 표시
                    timer_text = f"Timer: {elapsed:.1f}s / {self.required_duration}s"
                    cv2.putText(display_frame, timer_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 조건 유지 시간 충족 확인
                    if elapsed >= self.required_duration:
                        print(f"\n{'='*60}")
                        print(f"📸 조건이 {self.required_duration}초 이상 유지됨! 검사 시작...")
                        print(f"{'='*60}\n")
                        
                        # 카메라 종료
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        # 검사 수행
                        self._perform_inspection(self.last_valid_frame, self.last_valid_detections)
                        return
                else:
                    if self.condition_met:
                        # 조건이 해제됨
                        print(f"⚠️  조건 해제됨. 타이머 리셋.")
                        self.condition_met = False
                        self.condition_start_time = None
                        self.last_valid_frame = None
                        self.last_valid_detections = None
                    
                    # 상태 표시
                    status_text = "Waiting for condition..."
                    cv2.putText(display_frame, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 화면 표시
                cv2.imshow('Real-time Inspection', display_frame)
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n사용자가 종료함")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _check_condition(self, boxes):
        """조건 확인"""
        if boxes is None:
            # 검출 결과가 없는 경우
            if self.mode == 'frontdoor':
                return False, {'high': [], 'mid': [], 'low': []}
            else:  # bolt
                return False, {'bolts': [], 'frames': []}
        
        if self.mode == 'frontdoor':
            return self._check_frontdoor_condition(boxes)
        else:  # bolt
            return self._check_bolt_condition(boxes)
    
    def _check_frontdoor_condition(self, boxes):
        """프론트도어 조건 확인: high/mid/low 각 1개씩 OR high/low 각 1개씩"""
        detections = {'high': [], 'mid': [], 'low': []}
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if self.use_obb and hasattr(box, 'obb'):
                # OBB 모드: obb 속성 사용
                xyxyxyxy = box.obb.xyxyxyxy[0].cpu().numpy()
                # OBB를 일반 bbox로 변환 (시각화용)
                x_coords = xyxyxyxy[::2]
                y_coords = xyxyxyxy[1::2]
                xyxy = np.array([x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()])
                bbox = xyxyxyxy  # 실제 crop에는 8개 점 사용
            else:
                xyxy = box.xyxy[0].cpu().numpy()
                bbox = xyxy
            
            class_name = self.yolo_model.names[cls_id].lower()
            if class_name in detections:
                detections[class_name].append({
                    'bbox': bbox,
                    'conf': conf,
                    'cls_id': cls_id
                })
        
        # 조건: high/mid/low 각 1개씩 OR high/low 각 1개씩
        has_all_three = (len(detections['high']) == 1 and 
                        len(detections['mid']) == 1 and 
                        len(detections['low']) == 1)
        has_high_low = (len(detections['high']) == 1 and 
                       len(detections['low']) == 1 and 
                       len(detections['mid']) == 0)
        
        condition_met = has_all_three or has_high_low
        
        return condition_met, detections
    
    def _check_bolt_condition(self, boxes):
        """볼트 조건 확인: 2~7번 프레임 객체 정확히 1개"""
        bolt_detections = []  # 0, 1번 (볼트)
        frame_detections = []  # 2~7번 (프레임)
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if self.use_obb and hasattr(box, 'obb'):
                # OBB 모드: obb 속성 사용
                xyxyxyxy = box.obb.xyxyxyxy[0].cpu().numpy()
                # OBB의 경우 중심점 계산 (4개 점의 평균)
                center = [xyxyxyxy[::2].mean(), xyxyxyxy[1::2].mean()]
                bbox = xyxyxyxy
            else:
                xyxy = box.xyxy[0].cpu().numpy()
                center = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
                bbox = xyxy
            
            detection = {
                'class_id': cls_id,
                'bbox': bbox,
                'conf': conf,
                'center': center
            }
            
            if cls_id in [0, 1]:  # 볼트
                bolt_detections.append(detection)
            elif cls_id in [2, 3, 4, 5, 6, 7]:  # 프레임
                frame_detections.append(detection)
        
        # 조건: 프레임 객체 정확히 1개
        condition_met = len(frame_detections) == 1
        
        detections = {
            'bolts': bolt_detections,
            'frames': frame_detections
        }
        
        return condition_met, detections
    
    def _draw_detections(self, frame, boxes):
        """검출 결과를 프레임에 그리기"""
        if boxes is None:
            return frame
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 클래스명
            class_name = self.yolo_model.names[cls_id]
            
            # 색상 결정
            if self.mode == 'frontdoor':
                color = (0, 255, 0) if class_name.lower() in ['high', 'mid', 'low'] else (128, 128, 128)
            else:  # bolt
                if cls_id in [0, 1]:
                    color = (255, 0, 0)  # 파란색 (볼트)
                elif cls_id in [2, 3, 4, 5, 6, 7]:
                    color = (0, 255, 0)  # 초록색 (프레임)
                else:
                    color = (128, 128, 128)
            
            # OBB 모드인 경우 회전된 박스 그리기
            if self.use_obb and hasattr(box, 'obb'):
                xyxyxyxy = box.obb.xyxyxyxy[0].cpu().numpy()
                # 4개 점으로 변환
                points = np.array([
                    [xyxyxyxy[0], xyxyxyxy[1]],
                    [xyxyxyxy[2], xyxyxyxy[3]],
                    [xyxyxyxy[4], xyxyxyxy[5]],
                    [xyxyxyxy[6], xyxyxyxy[7]]
                ], dtype=np.int32)
                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
                x1, y1 = int(points[0][0]), int(points[0][1])
            else:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 라벨
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def _perform_inspection(self, frame, detections):
        """검사 수행"""
        if self.mode == 'frontdoor':
            self._inspect_frontdoor(frame, detections)
        else:  # bolt
            self._inspect_bolt(frame, detections)
    
    def _inspect_frontdoor(self, frame, detections):
        """프론트도어 검사"""
        print(f"🔍 프론트도어 검사 중...\n")
        
        part_results = {}
        parts_to_process = []
        
        # 처리할 부위 결정 (high/mid/low 또는 high/low)
        if len(detections['high']) == 1 and len(detections['mid']) == 1 and len(detections['low']) == 1:
            parts_to_process = ['high', 'mid', 'low']
        elif len(detections['high']) == 1 and len(detections['low']) == 1 and len(detections['mid']) == 0:
            parts_to_process = ['high', 'low']
        
        for part in parts_to_process:
            if len(detections[part]) > 0:
                bbox = detections[part][0]['bbox']
                
                # OBB 모드인 경우 회전된 객체 crop
                if self.use_obb and len(bbox) == 8:
                    cropped = self._crop_obb_object(frame, bbox)
                else:
                    x1, y1, x2, y2 = map(int, bbox)
                    cropped = frame[y1:y2, x1:x2]
                
                if cropped is None or cropped.size == 0:
                    print(f"  [{part.upper()}] 크롭 실패")
                    continue
                
                # DINOv2 분류
                result = self._classify_with_dino(cropped, part)
                
                part_results[part] = result
                
                # 출력 메시지
                if result['num_classes'] == 4:
                    result_text = "양품" if not result['is_defect'] else f"불량(클래스 {result['pred_class']})"
                    conf_display = result['confidence'][result['pred_class']]
                else:
                    result_text = "양품" if not result['is_defect'] else "불량"
                    conf_display = result['confidence'][result['pred_class']]
                
                print(f"  [{part.upper()}] {result_text} (신뢰도: {conf_display:.2%})")
        
        # Voting
        print(f"\n📊 최종 판정 ({self.voting_method.upper()} Voting):")
        if self.voting_method == 'hard':
            final_result = self._hard_voting(part_results)
        else:  # soft
            final_result = self._soft_voting(part_results)
        
        print(f"  결과: {'✅ 양품' if final_result == 'good' else '❌ 불량'}")
        print(f"\n{'='*60}\n")
    
    def _inspect_bolt(self, frame, detections):
        """볼트 검사"""
        print(f"🔍 볼트 검사 중...\n")
        
        frame_obj = detections['frames'][0]
        frame_bbox = frame_obj['bbox']
        frame_cls = frame_obj['class_id']
        
        # 프레임 클래스명
        frame_name = self.bolt_class_names.get(frame_cls, 'unknown')
        
        print(f"  프레임 타입: {frame_name}")
        
        # 프레임 내 볼트 찾기
        bolts_in_frame = []
        for bolt in detections['bolts']:
            cx, cy = bolt['center']
            # OBB 모드인 경우 bbox가 8개 점일 수 있음
            if self.use_obb and len(frame_bbox) == 8:
                # OBB의 경우 점이 프레임 내에 있는지 확인
                if self._point_in_obb(cx, cy, frame_bbox):
                    bolts_in_frame.append(bolt)
            else:
                # 일반 bbox
                if (frame_bbox[0] <= cx <= frame_bbox[2] and 
                    frame_bbox[1] <= cy <= frame_bbox[3]):
                    bolts_in_frame.append(bolt)
        
        print(f"  프레임 내 볼트 개수: {len(bolts_in_frame)}")
        
        # 2, 3, 4번 프레임: 볼트 2개 체크 (sedan, suv, hood)
        if frame_cls in [2, 3, 4]:
            if len(bolts_in_frame) != 2:
                print(f"\n📊 최종 판정:")
                print(f"  결과: ❌ 불량 (볼트 개수 불일치: {len(bolts_in_frame)}/2)")
                print(f"\n{'='*60}\n")
                return
        
        # 볼트가 없으면 불량
        if len(bolts_in_frame) == 0:
            print(f"\n📊 최종 판정:")
            print(f"  결과: ❌ 불량 (프레임 내 볼트 없음)")
            print(f"\n{'='*60}\n")
            return
        
        # 각 볼트 검사
        print(f"\n  볼트별 검사:")
        bolt_results = []
        for i, bolt in enumerate(bolts_in_frame):
            bbox = bolt['bbox']
            
            # OBB 모드인 경우 회전된 객체 crop
            if self.use_obb and len(bbox) == 8:
                cropped = self._crop_obb_object(frame, bbox)
            else:
                x1, y1, x2, y2 = map(int, bbox)
                cropped = frame[y1:y2, x1:x2]
            
            if cropped is None or cropped.size == 0:
                print(f"    볼트 #{i+1}: 크롭 실패")
                continue
            
            result = self._classify_with_dino(cropped, 'bolt')
            bolt_results.append(result)
            
            # 출력 메시지 (볼트는 항상 2-class)
            result_text = "양품" if not result['is_defect'] else "불량"
            conf_display = result['confidence'][result['pred_class']]
            
            print(f"    볼트 #{i+1}: {result_text} (신뢰도: {conf_display:.2%})")
        
        # Voting 방식으로 최종 판정
        print(f"\n📊 최종 판정 ({self.voting_method.upper()} Voting):")
        if self.voting_method == 'hard':
            final_result = self._hard_voting_bolt(bolt_results)
        else:  # soft
            final_result = self._soft_voting_bolt(bolt_results)
        
        print(f"  결과: {'✅ 양품' if final_result == 'good' else '❌ 불량'}")
        print(f"\n{'='*60}\n")
    
    def _classify_with_dino(self, cropped_img, part):
        """DINOv2로 분류"""
        # 볼트는 항상 2-class, 프론트도어만 4-class 가능
        is_bolt = (part == 'bolt')
        num_classes = 2 if is_bolt else self.dino_num_classes.get(part, 2)
        
        if cropped_img.size == 0:
            # 빈 이미지는 불량으로 처리
            if num_classes == 4:
                confidence = [0.0, 0.0, 0.0, 1.0]  # 클래스 3에 높은 confidence
                defect_confidence = 1.0
                pred_class = 3
            else:
                confidence = [0.0, 1.0]
                defect_confidence = 1.0
                pred_class = 1
            return {
                'is_defect': True,
                'confidence': confidence,
                'pred_class': pred_class,
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
        if num_classes == 4:
            # 프론트도어 4-class 모드: 0=양품, 1,2,3=불량
            is_defect = (pred_class != 0)
            # 불량 클래스들의 confidence 합계 계산 (소프트 보팅용)
            defect_confidence = sum(confidence[1:4]) if len(confidence) >= 4 else confidence[1] if len(confidence) >= 2 else 0.0
        else:
            # 2-class 모드 (볼트 또는 프론트도어 simple)
            is_defect = (pred_class == 1)
            defect_confidence = confidence[1] if len(confidence) >= 2 else 0.0
        
        # 반환: (양불량 판정, confidence 리스트, 원본 예측 클래스, 불량 confidence)
        return {
            'is_defect': is_defect,
            'confidence': confidence,
            'pred_class': pred_class,
            'defect_confidence': defect_confidence,
            'num_classes': num_classes
        }
    
    def _hard_voting(self, part_results):
        """Hard Voting: 0이 아니면 불량"""
        # 4-class 모드: 0이 아니면 불량
        # 2-class 모드: 1이면 불량
        has_defect = any(result['is_defect'] for result in part_results.values())
        return 'defect' if has_defect else 'good'
    
    def _soft_voting(self, part_results):
        """Soft Voting: 불량 confidence 평균"""
        if len(part_results) == 0:
            return 'good'
        
        # 각 부위의 불량 confidence 평균
        defect_confidences = [result['defect_confidence'] for result in part_results.values()]
        avg_defect_conf = sum(defect_confidences) / len(defect_confidences)
        
        # 평균이 0.5 이상이면 불량
        if avg_defect_conf >= 0.5:
            return 'defect'
        else:
            return 'good'
    
    def _hard_voting_bolt(self, bolt_results):
        """Hard Voting for Bolt: 하나라도 불량이면 불량 (0이 아니면 불량)"""
        if len(bolt_results) == 0:
            return 'good'
        
        has_defect = any(b['is_defect'] for b in bolt_results)
        return 'defect' if has_defect else 'good'
    
    def _soft_voting_bolt(self, bolt_results):
        """Soft Voting for Bolt: 평균 불량 confidence"""
        if len(bolt_results) == 0:
            return 'good'
        
        # 각 볼트의 불량 confidence 평균
        defect_confidences = [b['defect_confidence'] for b in bolt_results]
        avg_defect_conf = sum(defect_confidences) / len(defect_confidences)
        
        # 평균이 0.5 이상이면 불량
        if avg_defect_conf >= 0.5:
            return 'defect'
        else:
            return 'good'
    
    def _point_in_obb(self, x, y, obb_points):
        """점이 OBB 내부에 있는지 확인 (Ray casting algorithm)"""
        if len(obb_points) != 8:
            return False
        
        # 4개 점으로 변환
        points = [(obb_points[i], obb_points[i+1]) for i in range(0, 8, 2)]
        n = len(points)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = points[i]
            xj, yj = points[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _compute_rotated_box_corners(self, cx, cy, w, h, angle):
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
    
    def _correct_orientation_constrained(self, w, h, angle):
        """
        형상 적응형 보정 (Shape-Adaptive)
        조건: 객체는 원래 방향(가로/세로)에서 +-45도 이내로만 기울어짐.
        """
        pi = math.pi
        
        # 1. 각도 1차 정규화 (-pi ~ +pi)
        angle = (angle + pi) % (2 * pi) - pi
        
        # 2. 객체 형태에 따른 방향 보정
        if w >= h:
            # 가로가 긴 객체
            if abs(angle) > pi / 2:
                angle -= pi
        else:
            # 세로가 긴 객체
            if angle > 0:
                angle -= pi
            if angle < -pi + (pi/4):
                angle += pi
        
        # 최종 각도 재정규화
        angle = (angle + pi) % (2 * pi) - pi
        
        return w, h, angle
    
    def _crop_obb_object(self, img, obb_points):
        """
        OBB 좌표로부터 회전된 객체를 crop
        obb_points: [x1, y1, x2, y2, x3, y3, x4, y4] 형식
        """
        if len(obb_points) != 8:
            return None
        
        img_h, img_w = img.shape[:2]
        
        # 4개 점으로 변환
        points = np.array([
            [obb_points[0], obb_points[1]],
            [obb_points[2], obb_points[3]],
            [obb_points[4], obb_points[5]],
            [obb_points[6], obb_points[7]]
        ], dtype=np.float32)
        
        # 중심점과 크기 계산
        cx = points[:, 0].mean()
        cy = points[:, 1].mean()
        
        # 너비와 높이 계산 (첫 번째와 두 번째 점 사이의 거리)
        w = np.linalg.norm(points[1] - points[0])
        h = np.linalg.norm(points[2] - points[1])
        
        # 각도 계산
        vx = points[1][0] - points[0][0]
        vy = points[1][1] - points[0][1]
        angle = math.atan2(vy, vx)
        
        # 방향 보정
        w, h, angle = self._correct_orientation_constrained(w, h, angle)
        
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
        
        # 회전된 박스 crop
        src_corners = self._compute_rotated_box_corners(cx, cy, w, h, angle)
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


def load_config(config_path):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    required_keys = ['mode', 'yolo_model']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"설정 파일에 '{key}' 필드가 없습니다")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='실시간 카메라 양불량 검사 시스템')
    
    parser.add_argument('--config', type=str, required=True,
                        help='설정 YAML 파일 경로')
    parser.add_argument('--source', type=str, default='0',
                        help='카메라 소스 (0: 웹캠, RTSP URL 등, 기본값: 0)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='디바이스 (기본값: cuda)')
    parser.add_argument('--obb', action='store_true',
                        help='OBB(Oriented Bounding Box) 모드 사용')
    
    args = parser.parse_args()
    
    # 설정 파일 로드
    config = load_config(args.config)
    
    mode = config['mode'].lower()
    yolo_model = config['yolo_model']
    conf_threshold = config.get('conf_threshold', 0.25)
    dino_mode = config.get('dino_mode', 'simple')  # config에서 mode 읽기
    
    # DINOv2 모델 설정
    dino_models = {}
    if mode == 'frontdoor':
        dino_models = {
            'high': config['dino_high'],
            'mid': config['dino_mid'],
            'low': config['dino_low']
        }
        voting_method = config.get('voting_method', 'soft')
    else:  # bolt
        dino_models = {
            'bolt': config['dino_bolt']
        }
        voting_method = config.get('voting_method', 'soft')
    
    # 카메라 소스 처리
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # 시스템 초기화
    system = RealtimeInspectionSystem(
        mode=mode,
        yolo_model_path=yolo_model,
        dino_models=dino_models,
        device=args.device,
        conf_threshold=conf_threshold,
        voting_method=voting_method,
        use_obb=args.obb
    )
    
    # DINO 모드 설정 (config에서 읽은 값)
    system.dino_mode = dino_mode
    
    # 실행
    system.run(source=source)


if __name__ == "__main__":
    # 예시 1: 프론트도어 검사 (웹캠)
    # python realtime_inspection.py --config configs/frontdoor_realtime.yaml --source 0
    
    # 예시 2: 볼트 검사 (외부 카메라)
    # python realtime_inspection.py --config configs/bolt_realtime.yaml --source 1
    
    # 예시 3: RTSP 카메라
    # python realtime_inspection.py --config configs/frontdoor_realtime.yaml --source "rtsp://192.168.1.100:554/stream"
    
    main()