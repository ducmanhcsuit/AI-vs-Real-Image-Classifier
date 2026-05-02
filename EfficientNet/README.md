# 📷 AI vs. Real Image Classifier (Phân loại Ảnh Thực tế và Ảnh AI tạo sinh)

Hệ thống Web Demo ứng dụng Học sâu (Deep Learning) để phân biệt ảnh thực tế và ảnh do Trí tuệ nhân tạo (AI) tạo sinh. Đây là đồ án kết thúc môn học **Nhập môn Thị giác máy tính (CS231.Q22)**.

## 📖 Giới thiệu Dự án

Trong bối cảnh công nghệ Trí tuệ nhân tạo phát triển không ngừng, sự bùng nổ của các mô hình sinh ảnh (Generative Models) như GANs và Diffusion Models đã tạo ra một thách thức lớn đối với việc xác thực nội dung số. Những hình ảnh nhân tạo hiện nay đạt đến độ chân thực cực cao, khiến mắt người khó có thể phân biệt được.

Dự án này giải quyết bài toán phân loại nhị phân (Binary Image Classification) nhằm phát hiện và phân tách rõ ràng giữa tác phẩm nhiếp ảnh thực tế (Real) và hình ảnh do AI tạo sinh (Fake). 

Hệ thống ứng dụng phương pháp Transfer Learning, sử dụng kiến trúc mạng nơ-ron tích chập **EfficientNetB0** (Pre-trained trên tập dữ liệu ImageNet) làm lõi trích xuất đặc trưng đa tầng. Giao diện Web Demo trực quan được xây dựng hoàn toàn bằng framework **Streamlit**, cho phép người dùng tải ảnh lên và nhận đánh giá xác suất theo thời gian thực.

## 👥 Nhóm thực hiện & Giảng viên hướng dẫn

* **Giảng viên hướng dẫn:** [TS. Mai Tiến Dũng]
* **Thành viên 1:** [Trương Đức Mạnh] - [24521046]
* **Thành viên 2:** [Lâm Ngọc Quang Phúc] - [24521378]

## 🧠 Kiến trúc Mô hình & Dữ liệu

### Tập dữ liệu (Dataset)
Tập dữ liệu được thu thập và phân loại theo 5 chủ đề thực tế:
* `Animals`: Ảnh về động vật.
* `City`: Ảnh phong cảnh đô thị và kiến trúc.
* `Food`: Ảnh về ẩm thực và các món ăn.
* `Nature`: Ảnh về thiên nhiên, phong cảnh.
* `People`: Ảnh về con người và chân dung.

*(Mỗi nhóm dữ liệu đều bao gồm cả ảnh chụp thực tế và ảnh tương ứng do các mô hình AI tạo ra)*. Toàn bộ dữ liệu được tự động chia tỷ lệ chuẩn để huấn luyện và kiểm định: **70% Train - 15% Validation - 15% Test**.

### Chi tiết Mô hình
* **Backbone:** EfficientNetB0 (Pre-trained weights từ ImageNet, đóng băng các lớp cơ sở).
* **Kiến trúc tùy chỉnh:** Bổ sung `GlobalAveragePooling2D`, lớp `Dense` (256 units, ReLU) và `Dropout` (0.3) để chống quá khớp (Overfitting).
* **Input Shape:** (224, 224, 3).
* **Output:** Hàm Sigmoid cho bài toán phân loại nhị phân (0: AI_Generated, 1: Real).
* **Đánh giá Mô hình:** Đánh giá toàn diện trên tập Test thông qua các chỉ số: Độ chính xác (Accuracy), Độ chuẩn xác (Precision), Độ thu hồi (Recall), Điểm F1 (F1-Score) và Ma trận nhầm lẫn (Confusion Matrix).
* **Framework:** TensorFlow / Keras 3.
