# Phát hiện đối tượng giao thông với RetinaNet và Deformable DETR

Dự án phát hiện **11 lớp đối tượng giao thông** trên ảnh và video. Nhóm so sánh RetinaNet với hai backbone ResNet-50 và MobileNetV3, cùng Deformable DETR với backbone ResNet-50. Mỗi kiến trúc được thử nghiệm trên hai phiên bản dữ liệu để khảo sát ảnh hưởng của mất cân bằng lớp.

## Chạy demo Streamlit

Demo sử dụng các trọng số đã xuất sẵn trong [`App/Model`](./22022001-22022012-22022014/App/Model). RetinaNet chạy qua ONNX Runtime. Deformable DETR được nạp từ thư mục mô hình Hugging Face có tệp `model.safetensors` và cấu hình đi kèm.

### Cài đặt trên Windows

Chuẩn bị Git, [Git LFS](https://git-lfs.com/) và Python 3.11. Chạy các lệnh sau trong PowerShell.

```powershell
git lfs install
git clone --depth 1 https://github.com/trog-dev/Traffic-Detection-RetinaNet-DETR.git
cd Traffic-Detection-RetinaNet-DETR
git lfs pull
cd .\22022001-22022012-22022014\App

py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install streamlit opencv-python numpy onnxruntime Pillow "transformers==4.53.3" torch torchvision safetensors timm
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

Khởi chạy từ thư mục `App` vì [`app.py`](./22022001-22022012-22022014/App/app.py) đọc trọng số bằng đường dẫn tương đối `./Model/...`. Lệnh trên cài ONNX Runtime cho CPU. Nếu dùng NVIDIA GPU để chạy RetinaNet, thay gói `onnxruntime` bằng `onnxruntime-gpu` tương thích với môi trường CUDA. Các phụ thuộc của dự án được liệt kê trong [`requirements.txt`](./22022001-22022012-22022014/App/requirements.txt). [Hướng dẫn ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-python)

### Sử dụng demo

1. Mở `http://localhost:8501` và chọn **Balanced Dataset** hoặc **Imbalanced Dataset** để dùng trọng số của phiên bản dữ liệu tương ứng.
2. Chọn mô hình, sau đó tải ảnh **JPG, JPEG, PNG** hoặc video **MP4, AVI, MOV**.
3. Điều chỉnh **Confidence Threshold** để lọc dự đoán. Có thể bật **Weighted Boxes Fusion (WBF)** để gộp các khung dự đoán chồng lấn. Với video, dùng **Xử lý mỗi N frame** để điều chỉnh tần suất suy luận.

Nếu tệp `.onnx` hoặc `.safetensors` chỉ chứa dòng `version https://git-lfs.github.com/spec/v1` cùng mã `oid`, hãy chạy lại `git lfs pull` ở thư mục gốc của repo để tải trọng số.

## Dữ liệu và notebook

Theo báo cáo, tập dữ liệu được tổng hợp từ **Intersection-Flow-5K**, **Object Detection of 35 Classes**, **Vietnamese Traffic Sign Detection** và **Road Issues Detection**. Quy trình cân bằng kết hợp giảm mẫu ở lớp phổ biến, tăng tần suất lấy mẫu ở lớp hiếm và tăng cường dữ liệu bằng Mosaic. Tỷ lệ số đối tượng giữa lớp lớn nhất và nhỏ nhất được báo cáo giảm từ **327:1** xuống **15:1**.

Các nhãn trong ứng dụng gồm `Vehicle`, `Bus`, `Bicycle`, `Person`, `Engine`, `Truck`, `Tricycle`, `Obstacle`, `Pothole`, `Traffic Light` và `Traffic Sign`.

| Mô hình | Dữ liệu cân bằng | Dữ liệu mất cân bằng |
| --- | --- | --- |
| RetinaNet + ResNet-50 | [retinanet.ipynb](./22022001-22022012-22022014/retinanet.ipynb) | [retinanet-imbalance.ipynb](./22022001-22022012-22022014/retinanet-imbalance.ipynb) |
| RetinaNet + MobileNetV3 | [mobilenetv3.ipynb](./22022001-22022012-22022014/mobilenetv3.ipynb) | [imbalanced-retinanet-mobilenetv3.ipynb](./22022001-22022012-22022014/imbalanced-retinanet-mobilenetv3.ipynb) |
| Deformable DETR + ResNet-50 | [DETR-canbang.ipynb](./22022001-22022012-22022014/DETR-canbang.ipynb) | [DETR-matcanbang.ipynb](./22022001-22022012-22022014/DETR-matcanbang.ipynb) |

### Huấn luyện trên Kaggle

1. Bật GPU và thêm bộ dữ liệu đã chuẩn bị theo định dạng **COCO**. Các notebook dùng thư mục ảnh `train/`, `val/` và các tệp `annotations/instances_train.json`, `annotations/instances_val.json`. Các notebook RetinaNet còn dùng `test/` và `annotations/instances_test.json` ở bước kiểm thử.
2. Mở notebook tương ứng và cập nhật đường dẫn ảnh, annotation trong `CONFIG` theo thư mục thực tế dưới `/kaggle/input/`. Các đường dẫn gốc dùng `coco_split` cho dữ liệu cân bằng và `ImBalanced-Data` cho dữ liệu mất cân bằng.
3. Kiểm tra các đường dẫn checkpoint và log trước khi chạy. Để tiếp tục một lần huấn luyện, cung cấp checkpoint tương ứng. Khi bắt đầu lượt huấn luyện mới, bỏ qua cell sao chép checkpoint và log cũ trong notebook DETR hoặc tắt `RESUME_TRAINING` trong notebook RetinaNet. Notebook MobileNetV3 cân bằng còn tham chiếu `PRETRAINED_WEIGHTS` để khởi tạo từ trọng số của lần huấn luyện trước.

## Kết quả thực nghiệm

Bảng 3 của [báo cáo dự án](./22022001-22022012-22022014/22022001-22022012-22022014.pdf) tổng hợp mAP trên tập kiểm thử theo kiến trúc. Báo cáo mô tả các tập kiểm thử riêng cho hai phiên bản dữ liệu.

| Mô hình | mAP (%) |
| --- | ---: |
| RetinaNet + ResNet-50 | 36,52 |
| RetinaNet + MobileNetV3 | **38,53** |
| Deformable DETR + ResNet-50 | 34,74 |

Biểu đồ huấn luyện được lưu cùng dự án. Xem ví dụ [mAP theo epoch của RetinaNet + MobileNetV3 trên dữ liệu cân bằng](./22022001-22022012-22022014/MobileNetV3-balanced/fig_08_map_per_epoch.png).

## Nhóm thực hiện

Lê Nguyễn Thành Tài, Nguyễn Đức Trọng và Võ Lê Toàn Thắng. Khoa Công nghệ Thông tin, Trường Đại học Sư phạm Kỹ thuật Vĩnh Long.
