# 📷 AI vs. Real Image Classifier (Phân loại Ảnh Thực tế và Ảnh AI tạo sinh)

Hệ thống Web Demo ứng dụng Học sâu (Deep Learning) để phân biệt ảnh thực tế và ảnh do Trí tuệ nhân tạo (AI) tạo sinh. Đây là đồ án kết thúc môn học **Nhập môn Thị giác máy tính (CS231.Q22)**.

## 📖 Giới thiệu Dự án

Trong bối cảnh công nghệ Trí tuệ nhân tạo phát triển không ngừng, sự bùng nổ của các mô hình sinh ảnh (Generative Models) như GANs và Diffusion Models đã tạo ra một thách thức lớn đối với việc xác thực nội dung số. Những hình ảnh nhân tạo hiện nay đạt đến độ chân thực cực cao, khiến mắt người khó có thể phân biệt được.

Dự án này giải quyết bài toán phân loại nhị phân (Binary Image Classification) nhằm phát hiện và phân tách rõ ràng giữa tác phẩm nhiếp ảnh thực tế (Real) và hình ảnh do AI tạo sinh (Fake). 

Hệ thống ứng dụng phương pháp Transfer Learning, sử dụng kiến trúc mạng nơ-ron tích chập **MobileNetV2** (Pre-trained trên tập dữ liệu ImageNet) làm lõi trích xuất đặc trưng đa tầng. Giao diện Web Demo trực quan được xây dựng hoàn toàn bằng framework **Streamlit**, cho phép người dùng tải ảnh lên và nhận đánh giá xác suất theo thời gian thực.

## 👥 Nhóm thực hiện & Giảng viên hướng dẫn

* **Giảng viên hướng dẫn:** TS. Mai Tiến Dũng
* **Thành viên 1:** Trương Đức Mạnh - 24521046
* **Thành viên 2:** Lâm Ngọc Quang Phúc - 24521378

## 🧠 Kiến trúc Mô hình & Dữ liệu

### Tập dữ liệu (Dataset)
Tập dữ liệu được thu thập và phân loại theo 5 chủ đề thực tế:
* `Animals`: Ảnh về động vật.
* `City`: Ảnh phong cảnh đô thị và kiến trúc.
* `Food`: Ảnh về ẩm thực và các món ăn.
* `Nature`: Ảnh về thiên nhiên, phong cảnh.
* `People`: Ảnh về con người và chân dung.
*(Mỗi nhóm dữ liệu đều bao gồm cả ảnh chụp thực tế và ảnh tương ứng do các mô hình AI tạo ra)*

### Chi tiết Mô hình
* **Backbone:** MobileNetV2 (Pre-trained trên ImageNet).
* **Input Shape:** (224, 224, 3)
* **Output:** Softmax (2 phân lớp: AI_Generated và Real).
* **Đầu ra:** Mô hình dự đoán một giá trị xác suất $p \in [0, 1]$. Dựa trên ngưỡng quyết định, hệ thống gán nhãn Ảnh Thực tế hoặc Ảnh AI.
* **Framework:** TensorFlow / Keras 3.

## ⚙️ Hướng dẫn Cài đặt và Chạy Demo (Local)

**Bước 1: Clone kho lưu trữ về máy tính**
```bash
git clone <https://github.com/ducmanhcsuit/AI-vs-Real-Image-Classifier.git>
cd AI-Real-Image-Classification
```

**Bước 2: Tạo và kích hoạt môi trường ảo (Khuyên dùng)**
```bash
python -m venv venv

# Kích hoạt trên Windows:
venv\Scripts\activate

# Kích hoạt trên MacOS/Linux:
source venv/bin/activate
```

**Bước 3: Cài đặt các thư viện cần thiết**
```bash
pip install -r requirements.txt
```

**Bước 4: Khởi chạy ứng dụng Web**
```bash
streamlit run main.py
```

Trình duyệt sẽ tự động mở trang web tại địa chỉ http://localhost:8501.

🛠️ Công nghệ Sử dụng
Ngôn ngữ lập trình: Python

Deep Learning Framework: TensorFlow 2.x, Keras 3

Giao diện Web: Streamlit

Xử lý ảnh: OpenCV, Pillow (PIL)

Tính toán ma trận: NumPy


