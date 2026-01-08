import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
import os
import warnings

# Import for Deformable DETR
try:
    from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
    import torch
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# =========================================================================
# CẤU HÌNH
# =========================================================================
# Danh sách các models hỗ trợ
AVAILABLE_MODELS = {
    # ========== BALANCED DATASET MODELS ==========
    "Balanced-RetinaNet-ResNet50": {
        "path": "./Model/Balanced/retinanet-resnet50.onnx",
        "input_size": 432,  # ✅ Sửa từ 448 -> 432
        "description": "RetinaNet ResNet50 - Balanced Dataset",
        "type": "onnx",
        "category": "balanced"
    },
    "Balanced-RetinaNet-MobileNetV3": {
        "path": "./Model/Balanced/Balanced-MobileNetV3.onnx",
        "input_size": 432,  # ✅ Sửa từ 448 -> 432
        "description": "RetinaNet MobileNetV3 - Balanced Dataset",
        "type": "onnx",
        "category": "balanced"
    },
    "Balanced-RetinaNet-MobileNetV3-Pretrained": {
        "path": "./Model/Balanced/RetinaNet-MobileNetV3-Pretrained.onnx",
        "input_size": 432,  # ✅ Sửa từ 448 -> 432
        "description": "RetinaNet MobileNetV3 Pretrained - Balanced Dataset",
        "type": "onnx",
        "category": "balanced"
    },
    "Balanced-Deformable-DETR": {
        "path": "./Model/Balanced/Deformable-DETR",
        "input_size": 448,  # DETR giữ nguyên 448
        "description": "Deformable DETR - Balanced Dataset",
        "type": "detr",
        "category": "balanced"
    } if DETR_AVAILABLE else None,
    
    # ========== IMBALANCED DATASET MODELS ==========
    "Imbalanced-RetinaNet-ResNet50": {
        "path": "./Model/Imbalance/RetinaNet-ResNet50.onnx",
        "input_size": 432,  # ✅ Sửa từ 448 -> 432
        "description": "RetinaNet ResNet50 - Imbalanced Dataset",
        "type": "onnx",
        "category": "imbalanced"
    },
    "Imbalanced-RetinaNet-MobileNetV3": {
        "path": "./Model/Imbalance/Imbalance-MobileNetV3.onnx",
        "input_size": 432,  # ✅ Sửa từ 448 -> 432
        "description": "RetinaNet MobileNetV3 - Imbalanced Dataset",
        "type": "onnx",
        "category": "imbalanced"
    },
    "Imbalanced-Deformable-DETR": {
        "path": "./Model/Imbalance/imbalance-detr",
        "input_size": 448,  # DETR giữ nguyên 448
        "description": "Deformable DETR - Imbalanced Dataset",
        "type": "detr",
        "category": "imbalanced"
    } if DETR_AVAILABLE else None
}

# Remove None values
AVAILABLE_MODELS = {k: v for k, v in AVAILABLE_MODELS.items() if v is not None}

CLASSES = [
    '__background__', 'Vehicle', 'Bus', 'Bicycle', 'Person', 'Engine', 'Truck',
    'Tricycle', 'Obstacle', 'Pothole', 'Traffic Light', 'Traffic Sign'
]

# Màu sắc cho từng class
CLASS_COLORS = {
    'Vehicle': (255, 50, 50), 'Bus': (255, 128, 0), 'Bicycle': (0, 255, 255),
    'Person': (0, 255, 0), 'Engine': (128, 0, 128), 'Truck': (200, 100, 50),
    'Tricycle': (0, 128, 255), 'Obstacle': (255, 255, 0), 
    'Pothole': (255, 0, 255), 'Traffic Light': (0, 200, 200), 
    'Traffic Sign': (0, 0, 255), 'Default': (200, 200, 200)
}

def get_color(name):
    """Lấy màu cho class"""
    return CLASS_COLORS.get(name, CLASS_COLORS['Default'])

# =========================================================================
# XỬ LÝ MODEL
# =========================================================================
@st.cache_resource
def load_model(model_path, model_type="onnx"):
    """Load model (ONNX hoặc Deformable DETR) với GPU/CPU fallback"""
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy file: {model_path}")
        st.info("📥 Hãy đặt file model vào đúng thư mục")
        return None
    
    if model_type == "detr":
        try:
            # Load Deformable DETR từ safetensors
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = DeformableDetrForObjectDetection.from_pretrained(
                model_path,
                local_files_only=True
            ).to(device)
            model.eval()
            
            processor = DeformableDetrImageProcessor.from_pretrained(
                model_path,
                local_files_only=True
            )
            
            st.success(f"✅ Deformable DETR loaded on: {device.upper()}")
            # DETR thường dùng 448x448 hoặc lấy từ preprocessor config
            return {"model": model, "processor": processor, "device": device, "type": "detr", "input_size": 448}
        except Exception as e:
            st.error(f"❌ Lỗi load Deformable DETR: {e}")
            return None
    else:
        # Load ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            session = ort.InferenceSession(model_path, providers=providers)
            device = session.get_providers()[0]
            
            # ✅ Auto-detect input size từ model
            input_shape = session.get_inputs()[0].shape
            # ONNX shape có thể là [batch, channels, height, width] hoặc symbolic names
            # Cần lấy giá trị thực tế, không phải tên
            detected_size = 432  # Default fallback
            if len(input_shape) >= 3:
                # Thử lấy height (index 2 cho NCHW format)
                dim = input_shape[2]
                if isinstance(dim, int):
                    detected_size = dim
                elif isinstance(dim, str) and dim.isdigit():
                    detected_size = int(dim)
                # Nếu symbolic (e.g., 'height'), dùng fallback 432
            
            st.success(f"✅ Model loaded on: {device}")
            st.info(f"📐 Detected input size: {detected_size}x{detected_size}")
            
            return {
                "session": session, 
                "type": "onnx",
                "input_size": detected_size  # Trả về input size thực tế
            }
        except Exception as e:
            st.error(f"❌ Lỗi load model: {e}")
            return None

def preprocess(frame, input_size):
    """Preprocess ảnh cho model RetinaNet"""
    h, w = frame.shape[:2]
    scale = input_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(frame, (new_w, new_h))
    
    # Padding để đạt kích thước input_size x input_size
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w, :] = img_resized
    
    # Normalize theo ImageNet
    img = canvas.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # Chuyển từ HWC -> CHW và thêm batch dimension
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img, scale

def refine_boxes(boxes, scale, image_shape, min_box_size=10, max_aspect_ratio=10):
    """
    Tinh chỉnh và lọc boxes để có kết quả chuẩn hơn
    - Làm tròn tọa độ
    - Clip về biên ảnh
    - Loại bỏ boxes quá nhỏ
    - Loại bỏ boxes có tỷ lệ khung hình bất thường
    """
    refined = []
    h, w = image_shape[:2]
    
    for box in boxes:
        # Scale về kích thước gốc
        x1 = int(box[0] / scale)
        y1 = int(box[1] / scale)
        x2 = int(box[2] / scale)
        y2 = int(box[3] / scale)
        
        # Clip về biên ảnh
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Tính kích thước box
        box_w = x2 - x1
        box_h = y2 - y1
        
        # Loại bỏ boxes quá nhỏ
        if box_w < min_box_size or box_h < min_box_size:
            continue
            
        # Loại bỏ boxes có aspect ratio bất thường
        aspect_ratio = max(box_w, box_h) / max(min(box_w, box_h), 1)
        if aspect_ratio > max_aspect_ratio:
            continue
        
        refined.append([x1, y1, x2, y2])
    
    return refined

def calculate_iou(box1, box2):
    """Tính IoU (Intersection over Union) giữa 2 boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Tính diện tích giao
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Tính diện tích hợp
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thresh=0.55, conf_thresh=0.0):
    """
    Weighted Boxes Fusion - Merge multiple predictions into optimal boxes
    
    Chiến lược tối ưu:
    - Boxes CÙNG LỚP + overlap: Merge bằng WBF (weighted average)
    - Boxes KHÁC LỚP + overlap: Chọn box có confidence cao nhất
    
    Args:
        boxes_list: List of box arrays (có thể từ nhiều models hoặc TTA)
        scores_list: List of score arrays
        labels_list: List of label arrays
        iou_thresh: IoU threshold để merge boxes
        conf_thresh: Confidence threshold để filter
    """
    # Chuẩn hóa boxes về [0, 1]
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        for i in range(len(scores)):
            if scores[i] > conf_thresh:
                all_boxes.append(boxes[i])
                all_scores.append(scores[i])
                all_labels.append(int(labels[i]))
    
    if len(all_boxes) == 0:
        return [], [], []
    
    # Sort theo score giảm dần
    indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
    
    used = [False] * len(all_boxes)
    fused_boxes = []
    fused_scores = []
    fused_labels = []
    
    for idx in indices:
        if used[idx]:
            continue
        
        current_box = all_boxes[idx]
        current_score = all_scores[idx]
        current_label = all_labels[idx]
        
        # Tìm tất cả boxes overlap với box hiện tại
        same_class_cluster = [current_box]
        same_class_scores = [current_score]
        diff_class_competitors = []  # [(box, score, label, other_idx), ...]
        
        used[idx] = True
        
        for other_idx in indices:
            if used[other_idx] or other_idx == idx:
                continue
            
            iou = calculate_iou(current_box, all_boxes[other_idx])
            
            if iou >= iou_thresh:
                if all_labels[other_idx] == current_label:
                    # CÙNG LỚP: Thêm vào cluster để merge
                    same_class_cluster.append(all_boxes[other_idx])
                    same_class_scores.append(all_scores[other_idx])
                    used[other_idx] = True
                else:
                    # KHÁC LỚP: Ghi nhận để so sánh confidence
                    diff_class_competitors.append((
                        all_boxes[other_idx],
                        all_scores[other_idx],
                        all_labels[other_idx],
                        other_idx
                    ))
        
        # Kiểm tra xem có box khác lớp nào có confidence cao hơn không
        should_skip = False
        for comp_box, comp_score, comp_label, comp_idx in diff_class_competitors:
            if comp_score > current_score:
                # Box khác lớp có confidence cao hơn → Bỏ qua box hiện tại
                should_skip = True
                break
            else:
                # Box hiện tại thắng → Đánh dấu box khác lớp là đã dùng
                used[comp_idx] = True
        
        if should_skip:
            continue
        
        # Merge boxes cùng lớp bằng WBF
        if len(same_class_cluster) > 1:
            # Weighted average
            total_score = sum(same_class_scores)
            weights = [s / total_score for s in same_class_scores]
            
            x1 = sum(w * box[0] for w, box in zip(weights, same_class_cluster))
            y1 = sum(w * box[1] for w, box in zip(weights, same_class_cluster))
            x2 = sum(w * box[2] for w, box in zip(weights, same_class_cluster))
            y2 = sum(w * box[3] for w, box in zip(weights, same_class_cluster))
            
            # Score = trung bình có trọng số
            fused_score = total_score / len(same_class_scores)
            
            fused_boxes.append([x1, y1, x2, y2])
            fused_scores.append(fused_score)
            fused_labels.append(current_label)
        else:
            # Chỉ có 1 box → Giữ nguyên
            fused_boxes.append(current_box)
            fused_scores.append(current_score)
            fused_labels.append(current_label)
    
    return fused_boxes, fused_scores, fused_labels

def apply_tta(session, img_rgb, input_size, conf_thresh):
    """
    Test Time Augmentation - Dự đoán trên nhiều phiên bản ảnh
    - Original
    - Horizontal flip
    - Multi-scale (0.9x, 1.0x, 1.1x)
    
    Returns: List of (boxes, scores, labels) từ các augmentations
    """
    input_name = session.get_inputs()[0].name
    h_orig, w_orig = img_rgb.shape[:2]
    
    all_predictions = []
    
    # Các scales để test
    scales_to_test = [0.9, 1.0, 1.1]
    
    for scale_factor in scales_to_test:
        # Resize ảnh theo scale
        if scale_factor != 1.0:
            new_h = int(h_orig * scale_factor)
            new_w = int(w_orig * scale_factor)
            img_scaled = cv2.resize(img_rgb, (new_w, new_h))
        else:
            img_scaled = img_rgb.copy()
        
        h_scaled, w_scaled = img_scaled.shape[:2]
        
        # 1. Original
        img_in, scale = preprocess(img_scaled, input_size)
        outs = session.run(None, {input_name: img_in})
        boxes, scores, labels = outs[0], outs[1], outs[2]
        
        # Scale boxes về kích thước ảnh gốc (h_orig x w_orig)
        # Bước 1: Scale từ canvas (448x448) về img_scaled
        # Bước 2: Scale từ img_scaled về img_rgb gốc
        boxes_to_orig = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # Scale về img_scaled
            x1_scaled = x1 / scale
            y1_scaled = y1 / scale
            x2_scaled = x2 / scale
            y2_scaled = y2 / scale
            
            # Scale về img_rgb gốc
            if scale_factor != 1.0:
                x1_orig = x1_scaled / scale_factor
                y1_orig = y1_scaled / scale_factor
                x2_orig = x2_scaled / scale_factor
                y2_orig = y2_scaled / scale_factor
            else:
                x1_orig = x1_scaled
                y1_orig = y1_scaled
                x2_orig = x2_scaled
                y2_orig = y2_scaled
            
            boxes_to_orig.append([x1_orig, y1_orig, x2_orig, y2_orig])
        
        all_predictions.append((boxes_to_orig, scores, labels))
        
        # 2. Horizontal flip
        img_flipped = cv2.flip(img_scaled, 1)
        img_in_flip, scale_flip = preprocess(img_flipped, input_size)
        outs_flip = session.run(None, {input_name: img_in_flip})
        boxes_flip, scores_flip, labels_flip = outs_flip[0], outs_flip[1], outs_flip[2]
        
        # Flip boxes back và scale về gốc
        boxes_flip_to_orig = []
        for box in boxes_flip:
            x1, y1, x2, y2 = box
            
            # Scale về img_flipped (img_scaled size)
            x1_scaled = x1 / scale_flip
            y1_scaled = y1 / scale_flip
            x2_scaled = x2 / scale_flip
            y2_scaled = y2 / scale_flip
            
            # Flip coordinates back (đảo x)
            x1_unflipped = w_scaled - x2_scaled
            x2_unflipped = w_scaled - x1_scaled
            
            # Scale về img_rgb gốc
            if scale_factor != 1.0:
                x1_orig = x1_unflipped / scale_factor
                y1_orig = y1_scaled / scale_factor
                x2_orig = x2_unflipped / scale_factor
                y2_orig = y2_scaled / scale_factor
            else:
                x1_orig = x1_unflipped
                y1_orig = y1_scaled
                x2_orig = x2_unflipped
                y2_orig = y2_scaled
            
            boxes_flip_to_orig.append([x1_orig, y1_orig, x2_orig, y2_orig])
        
        all_predictions.append((boxes_flip_to_orig, scores_flip, labels_flip))
    
    return all_predictions

def infer_detr(model_dict, img_rgb, conf_thresh):
    """
    Inference với Deformable DETR model
    Returns: boxes, scores, labels (format giống ONNX để tương thích)
    
    DETR mapping: 0=Vehicle, 1=Bus, 2=Bicycle, 3=Person, 4=Engine, 5=Truck,
                  6=Tricycle, 7=Obstacle, 8=Pothole, 9=Traffic Light, 10=Traffic Sign
    RetinaNet mapping: 0=__background__, 1=Vehicle, 2=Bus, 3=Bicycle, 4=Person, 5=Engine,
                       6=Truck, 7=Tricycle, 8=Obstacle, 9=Pothole, 10=Traffic Light, 11=Traffic Sign
    """
    model = model_dict["model"]
    processor = model_dict["processor"]
    device = model_dict["device"]
    
    # Preprocess image
    pil_img = Image.fromarray(img_rgb)
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process
    target_sizes = torch.tensor([img_rgb.shape[:2]]).to(device)
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes,
        threshold=conf_thresh
    )[0]
    
    # Convert to numpy arrays
    boxes = results["boxes"].cpu().numpy()  # [x1, y1, x2, y2]
    scores = results["scores"].cpu().numpy()
    labels_detr = results["labels"].cpu().numpy()
    
    # Map DETR labels (0-10) to RetinaNet labels (1-11) để tương thích
    # DETR: 0=Vehicle → RetinaNet: 1=Vehicle
    # DETR: 1=Bus → RetinaNet: 2=Bus, v.v.
    detr_to_retinanet = {
        0: 1,   # Vehicle
        1: 2,   # Bus
        2: 3,   # Bicycle
        3: 4,   # Person
        4: 5,   # Engine
        5: 6,   # Truck
        6: 7,   # Tricycle
        7: 8,   # Obstacle
        8: 9,   # Pothole
        9: 10,  # Traffic Light
        10: 11  # Traffic Sign
    }
    
    # Convert labels
    labels = np.array([detr_to_retinanet.get(int(l), 0) for l in labels_detr])
    
    return boxes, scores, labels

def draw_detections(frame, boxes, scores, labels, conf_thresh, scale, iou_thresh=0.55, 
                   min_box_size=10, max_aspect_ratio=10, use_wbf=True, skip_refinement=False):
    """Vẽ bounding boxes lên frame sau khi áp dụng WBF và refinement"""
    # Áp dụng WBF để merge duplicate boxes (tốt hơn NMS)
    if use_wbf:
        wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
            [boxes], [scores], [labels], iou_thresh, conf_thresh
        )
    else:
        # Fallback: filter theo confidence
        valid_indices = [i for i in range(len(scores)) if scores[i] > conf_thresh]
        wbf_boxes = [boxes[i] for i in valid_indices]
        wbf_scores = [scores[i] for i in valid_indices]
        wbf_labels = [labels[i] for i in valid_indices]
    
    # Refine boxes (có thể bỏ qua để tăng tốc)
    if skip_refinement:
        # Fast path: Không refine, chỉ scale trực tiếp
        final_boxes = []
        for box in wbf_boxes:
            x1 = int(box[0] / scale)
            y1 = int(box[1] / scale)
            x2 = int(box[2] / scale)
            y2 = int(box[3] / scale)
            final_boxes.append([x1, y1, x2, y2])
        final_scores = wbf_scores
        final_labels = wbf_labels
    else:
        # Normal path: Refine boxes để có kết quả chuẩn hơn
        refined_coords = refine_boxes(wbf_boxes, scale, frame.shape, min_box_size, max_aspect_ratio)
        
        # ✅ Filter lại sau refinement - ĐẢM BẢO SYNC
        # refined_coords có thể ngắn hơn wbf_boxes do filtering
        final_boxes = []
        final_scores = []
        final_labels = []
        
        # Map refined boxes về indices gốc
        refined_set = set()
        for refined_box in refined_coords:
            # Tìm box gốc tương ứng
            for orig_idx, orig_box in enumerate(wbf_boxes):
                if orig_idx in refined_set:
                    continue
                # So sánh tọa độ (sau khi scale)
                x1_orig = int(orig_box[0] / scale)
                y1_orig = int(orig_box[1] / scale)
                x2_orig = int(orig_box[2] / scale)
                y2_orig = int(orig_box[3] / scale)
                
                if (refined_box[0] == x1_orig and refined_box[1] == y1_orig and
                    refined_box[2] == x2_orig and refined_box[3] == y2_orig):
                    final_boxes.append(refined_box)
                    final_scores.append(wbf_scores[orig_idx])
                    final_labels.append(wbf_labels[orig_idx])
                    refined_set.add(orig_idx)
                    break
    
    count = 0
    for i in range(len(final_scores)):
        x1, y1, x2, y2 = final_boxes[i]
        
        cls_id = int(final_labels[i])
        name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
        color = get_color(name)
        
        # Thickness động dựa trên kích thước box
        box_area = (x2 - x1) * (y2 - y1)
        thickness = max(1, min(4, int(np.sqrt(box_area) / 50)))
        
        # Vẽ box với thickness động
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Vẽ label với background
        label_text = f"{name} {final_scores[i]:.2f}"
        font_scale = max(0.4, min(0.7, thickness * 0.2))
        font_thickness = max(1, thickness - 1)
        
        # Tính kích thước text
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Vẽ background cho text
        cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5), 
                     (x1 + text_w, y1), color, -1)
        
        # Vẽ text
        cv2.putText(frame, label_text, (x1, y1 - baseline - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        count += 1
    
    return frame, count

# =========================================================================
# UI CHÍNH
# =========================================================================
st.set_page_config(
    layout="wide", 
    page_title="Traffic AI - Multi Model",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main title styling */
    .main h1 {
        color: #1E88E5;
        font-size: 3rem !important;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1E88E5 0%, #42A5F5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    div[data-testid="metric-container"] > div {
        color: white !important;
    }
    
    div[data-testid="metric-container"] label {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: transparent;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #64B5F6 !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: #E3F2FD !important;
        font-weight: 500 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #42A5F5;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

st.title("🚦 Traffic AI - Multi Model Detection")

# Sidebar cho cấu hình
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình Model")
    st.markdown("---")
    
    # Nhóm models theo category
    balanced_models = {k: v for k, v in AVAILABLE_MODELS.items() if v and v.get('category') == 'balanced'}
    imbalanced_models = {k: v for k, v in AVAILABLE_MODELS.items() if v and v.get('category') == 'imbalanced'}
    
    # Chọn dataset type
    dataset_type = st.radio(
        "📊 Chọn Dataset Type",
        ["Balanced Dataset", "Imbalanced Dataset"],
        help="Balanced: Dữ liệu cân bằng các classes\nImbalanced: Dữ liệu mất cân bằng"
    )
    
    # Chọn model dựa trên dataset type
    if dataset_type == "Balanced Dataset":
        model_options = list(balanced_models.keys())
        st.success("✅ Balanced Dataset - Hiệu suất tốt cho tất cả classes")
    else:
        model_options = list(imbalanced_models.keys())
        st.warning("⚠️ Imbalanced Dataset - Có thể bias về class phổ biến")
    
    selected_model = st.selectbox(
        "Chọn Model Architecture",
        options=model_options,
        index=0
    )
    
    # Hiển thị thông tin model
    model_info = AVAILABLE_MODELS[selected_model]
    st.info(f"📝 {model_info['description']}")
    st.caption(f"🖼️ Input size: {model_info['input_size']}x{model_info['input_size']}")
    st.caption(f"⚙️ Type: {model_info['type'].upper()}")
    
    st.markdown("---")
    st.markdown("### 🎚️ Thông số Detection")
    
    # Confidence threshold
    conf_thresh = st.slider(
        "📊 Confidence Threshold", 
        0.0, 1.0, 0.45, 0.05,
        help="Ngưỡng độ tin cậy tối thiểu để hiển thị bbox"
    )
    
    # IoU threshold cho NMS
    iou_thresh = st.slider(
        "🔗 IoU Threshold (WBF)", 
        0.1, 0.9, 0.5, 0.05,
        help="IoU < threshold → Giữ box riêng, IoU ≥ threshold → Merge boxes"
    )
    
    st.markdown("---")
    # Advanced options
    with st.expander("🔧 Tùy chọn nâng cao", expanded=False):
        st.success("✨ WBF (Weighted Boxes Fusion) - Merge thông minh thay vì loại bỏ")
        
        use_wbf = st.checkbox(
            "🔄 Sử dụng WBF", 
            value=True,
            help="Merge boxes bằng weighted average - tốt hơn NMS"
        )
        
        st.markdown("**⚡ Tối ưu FPS**")
        skip_wbf_video = st.checkbox(
            "Bỏ qua WBF cho video (tăng tốc ~30%)",
            value=False,
            help="Chỉ filter theo confidence, không merge boxes"
        )
        
        reduce_display_quality = st.checkbox(
            "Giảm chất lượng hiển thị video",
            value=False,
            help="JPEG quality 60% thay vì 85% - tăng tốc ~15%"
        )
        
        skip_refinement = st.checkbox(
            "Bỏ qua Box Refinement (tăng tốc ~20%)",
            value=False,
            help="Không filter boxes nhỏ/bất thường - tăng tốc đáng kể"
        )
        
        ultra_mode = st.checkbox(
            "🚀 ULTRA MODE - Tăng tốc tối đa",
            value=False,
            help="Bật tất cả tối ưu: Bỏ WBF, giảm quality, bỏ refinement, resize nhỏ hơn"
        )
        
        st.markdown("**📏 Box Refinement**")
        min_box_size = st.slider(
            "Kích thước tối thiểu (px)", 
            5, 50, 10, 5,
            help="Lọc bỏ boxes quá nhỏ (nhiễu)"
        )
        
        max_aspect_ratio = st.slider(
            "Tỷ lệ khung hình tối đa", 
            3.0, 20.0, 10.0, 1.0,
            help="Lọc bỏ boxes có tỷ lệ dài/rộng bất thường"
        )
    
    st.markdown("---")
    # Hiển thị classes
    with st.expander("📋 Danh sách Classes (11 loại)", expanded=False):
        cols = st.columns(2)
        for i, cls in enumerate(CLASSES[1:], 1):  # Skip background
            col_idx = (i - 1) % 2
            with cols[col_idx]:
                st.markdown(f"**{i}.** {cls}")
    
    st.markdown("---")
    # Thông tin về Dataset Type
    with st.expander("ℹ️ Về Dataset Types", expanded=False):
        st.markdown("""
        **🟢 Balanced Dataset:**
        - Số lượng samples cân bằng giữa các classes
        - Hiệu suất tốt cho tất cả các đối tượng
        - Phù hợp cho production
        - AP cao hơn cho rare classes
        
        **🔴 Imbalanced Dataset:**
        - Dữ liệu mất cân bằng (thiên về một số classes)
        - Có thể bias về class phổ biến (Vehicle, Person)
        - Hiệu suất thấp hơn cho rare classes (Pothole, Tricycle)
        - Sử dụng để demo vấn đề imbalance
        
        **📊 So sánh:**
        - Balanced: mAP cao hơn 5-10%
        - Imbalanced: Tốc độ tương đương, nhưng kém chính xác
        """)

# Load model đã chọn
model_dict = load_model(model_info['path'], model_info['type'])
# ✅ Ưu tiên input_size từ model (auto-detected) hơn từ config
# Đảm bảo luôn là int, không phải string
if model_dict:
    detected_size = model_dict.get('input_size')
    input_size = int(detected_size) if detected_size is not None else int(model_info['input_size'])
else:
    input_size = int(model_info['input_size'])

if model_dict:
    # Hiển thị thông tin model đã load
    st.success(f"✅ Model loaded: **{selected_model}** ({dataset_type})")
    
    # Chọn chế độ xử lý
    mode = st.radio("Chế độ xử lý:", ["Video File", "Hình ảnh"], horizontal=True)

    # --- CHẾ ĐỘ VIDEO FILE ---
    if mode == "Video File":
        uploaded = st.file_uploader("Chọn video...", type=['mp4', 'avi', 'mov'])
        if uploaded:
            # Lưu video tạm
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as tfile:
                tfile.write(uploaded.read())
            
            vf = cv2.VideoCapture(temp_path)
            fps = int(vf.get(cv2.CAP_PROP_FPS))
            total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"Video: {total_frames} frames @ {fps} FPS | Model: {selected_model}")
            
            # Performance tips
            with st.expander("⚡ Tips tăng FPS", expanded=False):
                st.markdown("""
                **Cách tăng tốc độ xử lý:**
                
                **🚀 ULTRA MODE (Khuyến nghị):**
                - Bật "ULTRA MODE" trong Tùy chọn nâng cao → **tăng 5-10x FPS**
                - Tự động bật tất cả tối ưu hóa
                
                **Manual Settings:**
                1. 🎯 **Xử lý mỗi N frame** = 2-3 (tăng 2-3x FPS)
                2. 📉 **Độ rộng hiển thị** = 600px (giảm encoding time)
                3. ⚡ **Bỏ qua WBF** (+30% FPS)
                4. 🎨 **Giảm chất lượng JPEG** (+15% FPS)
                5. 📦 **Bỏ qua Box Refinement** (+20% FPS)
                6. 💻 **Dùng GPU** nếu có NVIDIA GPU
                7. 🏃 **Model nhẹ**: Balanced-MobileNetV3 (nhanh nhất)
                
                **Kết quả dự kiến:**
                - Normal: 3-5 FPS
                - Optimized: 10-15 FPS
                - **ULTRA MODE: 20-30 FPS** 🔥
                """)
            
            # Thêm options
            col1, col2 = st.columns(2)
            with col1:
                default_skip = 3 if ultra_mode else 1
                process_every_n = st.slider("Xử lý mỗi N frame", 1, 10, default_skip)
            with col2:
                default_width = 600 if ultra_mode else 800
                max_display_width = st.slider("Độ rộng hiển thị", 400, 1200, default_width, 100)
            
            # Apply ultra mode settings
            if ultra_mode:
                skip_wbf_video = True
                reduce_display_quality = True
                skip_refinement = True
                st.success("🚀 ULTRA MODE ACTIVE: Tốc độ tối đa!")
            
            # Buttons cho start/stop
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                start_btn = st.button("▶️ Bắt đầu xử lý", width="stretch")
            with col_btn2:
                stop_btn = st.button("⏸️ Dừng xử lý", width="stretch")
            
            # Initialize session state for stop flag
            if 'stop_processing' not in st.session_state:
                st.session_state.stop_processing = False
            
            # Set stop flag when stop button clicked
            if stop_btn:
                st.session_state.stop_processing = True
                st.warning("⚠️ Đang dừng xử lý...")
            
            if start_btn:
                st.session_state.stop_processing = False
                st_frame = st.empty()
                st_stats = st.empty()
                progress_bar = st.progress(0)
                
                # Stats display area
                stats_cols = st.columns(4)
                fps_display = stats_cols[0].empty()
                frame_display = stats_cols[1].empty()
                obj_display = stats_cols[2].empty()
                time_display = stats_cols[3].empty()
                
                frame_count = 0
                processed_count = 0
                start_time = time.time()
                last_time = start_time
                
                while vf.isOpened():
                    # Check stop flag
                    if st.session_state.stop_processing:
                        vf.release()  # ✅ Giải phóng resource
                        st.warning("⏸️ Đã dừng xử lý!")
                        break
                    
                    ret, frame = vf.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Chỉ xử lý mỗi N frames
                    if frame_count % process_every_n != 0:
                        continue
                    
                    # Tính FPS hiện tại
                    current_time = time.time()
                    instant_fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
                    last_time = current_time
                    
                    processed_count += 1
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Inference
                    if model_dict['type'] == 'detr':
                        boxes, scores, labels = infer_detr(model_dict, frame_rgb, conf_thresh)
                        scale = 1.0
                    else:
                        session = model_dict['session']
                        img_in, scale = preprocess(frame_rgb, input_size)
                        input_name = session.get_inputs()[0].name
                        outs = session.run(None, {input_name: img_in})
                        boxes, scores, labels = outs[0], outs[1], outs[2]
                    
                    # Vẽ detections với WBF tùy chọn (tắt WBF cho video để tăng tốc)
                    use_wbf_frame = use_wbf and not skip_wbf_video
                    frame_with_boxes, det_count = draw_detections(
                        frame_rgb.copy(), boxes, scores, labels, conf_thresh, scale, iou_thresh,
                        min_box_size, max_aspect_ratio, use_wbf_frame, skip_refinement
                    )
                    
                    # Hiển thị với nén JPEG tối ưu
                    h, w = frame_with_boxes.shape[:2]
                    display_w = min(w, max_display_width)
                    display_h = int(display_w * h / w)
                    
                    # Tối ưu resize: dùng INTER_NEAREST nếu ultra mode (nhanh hơn 2x)
                    interpolation = cv2.INTER_NEAREST if ultra_mode else cv2.INTER_LINEAR
                    frame_resized = cv2.resize(frame_with_boxes, (display_w, display_h), interpolation=interpolation)
                    
                    # Nén JPEG với quality tùy chọn
                    jpeg_quality = 60 if reduce_display_quality else 85
                    frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    _, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)
                    
                    st_frame.image(buffer.tobytes(), channels="BGR", width="stretch")
                    
                    # Hiển thị stats
                    elapsed = time.time() - start_time
                    avg_fps = processed_count / elapsed if elapsed > 0 else 0
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    
                    # Update metrics with current FPS
                    fps_display.metric("⚡ FPS hiện tại", f"{instant_fps:.1f}", delta=f"Avg: {avg_fps:.1f}")
                    frame_display.metric("🎞️ Frame", f"{frame_count}/{total_frames}", delta=f"{progress*100:.1f}%")
                    obj_display.metric("🎯 Objects", det_count)
                    time_display.metric("⏱️ Thời gian", f"{elapsed:.1f}s")
                
                vf.release()
                os.remove(temp_path)
                
                if not st.session_state.stop_processing:
                    st.success(f"✅ Hoàn thành! Xử lý {processed_count}/{frame_count} frames trong {elapsed:.1f}s (Avg FPS: {avg_fps:.1f})")
                else:
                    st.info(f"⏸️ Đã dừng tại frame {frame_count}/{total_frames}. Đã xử lý {processed_count} frames trong {elapsed:.1f}s")
                
                # Reset stop flag
                st.session_state.stop_processing = False

    
    # --- CHẾ ĐỘ HÌNH ẢNH ---
    elif mode == "Hình ảnh":
        # TTA option (chỉ cho ONNX models)
        enable_tta = False
        if model_dict['type'] == 'onnx':
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 🔬 Test Time Augmentation")
                enable_tta = st.checkbox(
                    "✨ Enable TTA", 
                    value=False,
                    help="Ensemble 6 augmentations để tăng độ chính xác"
                )
                if enable_tta:
                    st.warning("⏱️ Tăng accuracy ~3-5% nhưng chậm hơn **6x**")
                    st.info("📐 6 Augmentations:\n- Original (3 scales)\n- Horizontal Flip (3 scales)")
        
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #667eea;'>📤 Upload ảnh để phát hiện</h3>
            <p style='color: #666;'>Hỗ trợ: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader(
            "Chọn file ảnh", 
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        if uploaded:
            # Đọc ảnh
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            st.markdown("---")
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("### 📷 Ảnh gốc")
                st.image(img_rgb, width="stretch")
            
            # Inference với hoặc không có TTA
            if enable_tta and model_dict['type'] == 'onnx':
                with st.spinner(f"🔬 Đang phát hiện với TTA (6 augmentations)..."):
                    start_inference = time.time()
                    
                    # Apply TTA - trả về list of predictions
                    session = model_dict['session']
                    all_predictions = apply_tta(session, img_rgb, input_size, conf_thresh)
                    
                    # Unpack predictions
                    boxes_list = [pred[0] for pred in all_predictions]
                    scores_list = [pred[1] for pred in all_predictions]
                    labels_list = [pred[2] for pred in all_predictions]
                    
                    # WBF để merge tất cả predictions
                    boxes, scores, labels = weighted_boxes_fusion(
                        boxes_list, scores_list, labels_list, iou_thresh, conf_thresh
                    )
                    scale = 1.0  # TTA boxes đã ở kích thước gốc
                    
                    # Vẽ detections
                    img_result, det_count = draw_detections(
                        img_rgb.copy(), boxes, scores, labels, conf_thresh, scale, iou_thresh,
                        min_box_size, max_aspect_ratio, use_wbf
                    )
                    
                    inference_time = time.time() - start_inference
                    st.info(f"✅ TTA completed: {len(all_predictions)} augmentations merged")
            else:
                with st.spinner(f"Đang phát hiện với {selected_model}..."):
                    start_inference = time.time()
                    
                    # Check model type
                    if model_dict['type'] == 'detr':
                        # Deformable DETR inference
                        boxes, scores, labels = infer_detr(model_dict, img_rgb, conf_thresh)
                        scale = 1.0  # DETR trả về boxes ở kích thước gốc
                    else:
                        # ONNX inference
                        session = model_dict['session']
                        img_in, scale = preprocess(img_rgb, input_size)
                        input_name = session.get_inputs()[0].name
                        outs = session.run(None, {input_name: img_in})
                        boxes, scores, labels = outs[0], outs[1], outs[2]
                    
                    img_result, det_count = draw_detections(
                        img_rgb.copy(), boxes, scores, labels, conf_thresh, scale, iou_thresh,
                        min_box_size, max_aspect_ratio, use_wbf
                    )
                    inference_time = time.time() - start_inference
            
            with col2:
                st.markdown(f"### 🎯 Kết quả ({det_count} objects)")
                st.image(img_result, width="stretch")
            
            st.markdown("---")
            # Metrics with icons
            st.markdown("### 📊 Thống kê Performance")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(
                    "🎯 Objects Detected", 
                    det_count,
                    delta="TTA" if enable_tta else "Standard"
                )
            with col_m2:
                st.metric(
                    "⚡ Inference Time", 
                    f"{inference_time:.3f}s",
                    delta=f"{1000*inference_time:.0f}ms"
                )
            with col_m3:
                st.metric(
                    "🤖 Model", 
                    selected_model.split('-')[0],
                    delta=selected_model.split('-')[1] if '-' in selected_model else ""
                )
            
            # Hiển thị chi tiết detections
            if det_count > 0:
                st.markdown("---")
                st.markdown("### 📝 Chi tiết phát hiện")
                
                # Thống kê theo class
                class_count = {}
                for i in range(len(scores)):
                    if scores[i] > conf_thresh:
                        cls_id = int(labels[i])
                        name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
                        class_count[name] = class_count.get(name, 0) + 1
                
                # Hiển thị thống kê
                st.markdown("#### 📈 Phân bố theo Class")
                num_cols = min(len(class_count), 4)
                cols = st.columns(num_cols)
                for idx, (cls_name, count) in enumerate(class_count.items()):
                    with cols[idx % num_cols]:
                        # Icon mapping for common classes
                        icons = {
                            'Vehicle': '🚗', 'Bus': '🚌', 'Bicycle': '🚲',
                            'Person': '👤', 'Engine': '🏍️', 'Truck': '🚚',
                            'Tricycle': '🛺', 'Traffic Light': '🚦', 'Traffic Sign': '🚸'
                        }
                        icon = icons.get(cls_name, '📦')
                        st.metric(f"{icon} {cls_name}", count)
                
                # Bảng chi tiết
                st.markdown("#### 🔍 Chi tiết từng Detection")
                detection_data = []
                for i in range(len(scores)):
                    if scores[i] > conf_thresh:
                        cls_id = int(labels[i])
                        name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
                        box = boxes[i]
                        detection_data.append({
                            "#": len(detection_data) + 1,
                            "Class": name,
                            "Confidence": f"{scores[i]:.2%}",
                            "X1": int(box[0]),
                            "Y1": int(box[1]),
                            "X2": int(box[2]),
                            "Y2": int(box[3])
                        })
                st.dataframe(detection_data, width="stretch")
                
                # Download button cho kết quả
                st.markdown("---")
                st.markdown("### 💾 Tải về kết quả")
                result_bgr = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', result_bgr)
                
                col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                with col_dl2:
                    st.download_button(
                        label="📥 Download ảnh đã annotate",
                        data=buffer.tobytes(),
                        file_name=f"traffic_ai_{selected_model}_{int(time.time())}.jpg",
                        mime="image/jpeg",
                        width="stretch"
                    )
else:
    st.error("❌ Không thể load model. Vui lòng kiểm tra lại cấu hình.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <p><strong>🚦 Traffic AI - Multi Model Detection System</strong></p>
    <p style='font-size: 0.9em;'>Powered by RetinaNet | WBF Fusion | TTA Enhancement</p>
    <p style='font-size: 0.8em;'>🔄 WBF • 🔬 TTA • ⚡ Real-time Processing</p>
</div>
""", unsafe_allow_html=True)