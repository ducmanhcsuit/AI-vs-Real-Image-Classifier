import os
# Tắt cảnh báo C++ của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Tắt cảnh báo Python của TensorFlow
tf.get_logger().setLevel(logging.ERROR)

# --- 1. Tiêu đề và giao diện ---
st.set_page_config(page_title="Demo Phân loại ảnh", page_icon="📷")
st.title("Đồ án môn học Nhập môn thị giác máy tính - CS231.Q22")
st.header("🔍 Phân loại ảnh Thực tế và ảnh do AI tạo sinh")
st.write("Chào mừng bạn đến với trang web demo của đồ án. Vui lòng tải lên một bức ảnh để hệ thống kiểm tra!")

# --- 2. Tải mô hình ---
@st.cache_resource # Giúp load mô hình 1 lần duy nhất để web không bị giật lag
def load_model():
    # Lấy đường dẫn folder của file main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model_AI_Real.keras')
    
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy file model tại: {model_path}")
        st.stop()
    
    return tf.keras.models.load_model(model_path)

model = load_model()

# --- 3. Xử lý Upload Ảnh ---
uploaded_file = st.file_uploader("Chọn một bức ảnh (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mở và hiển thị ảnh
    image = Image.open(uploaded_file)
    
    # Căn giữa ảnh trên giao diện web
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Ảnh bạn vừa tải lên', use_container_width=True)
    
    st.write("---")
    
    # --- 4. Xử lý Dự đoán với MobileNetV2 ---
    if st.button('🚀 Phân tích ảnh này', use_container_width=True):
        with st.spinner("Hệ thống AI đang phân tích đa tầng..."):
            
            # XỬ LÝ ẢNH ĐẦU VÀO
            # Nếu ảnh là PNG trong suốt (RGBA) hoặc ảnh xám (L), chuyển hết về hệ màu chuẩn RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # BƯỚC QUAN TRỌNG ĐÃ ĐƯỢC THÊM: Đưa kích thước ảnh về chuẩn 224x224
            img_resized = image.resize((224, 224))
            
            # Đưa ảnh về mảng numpy và chuẩn hóa giống cách huấn luyện với MobileNetV2
            img_array = np.array(img_resized).astype('float32')
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # CHẠY DỰ ĐOÁN
            # Mô hình xuất ra mảng 2 phần tử tương ứng với 2 class
            prediction_prob = model.predict(img_array)[0]
            
            # Lưu ý: Sắp xếp vị trí 0 và 1 này phải khớp đúng với lúc bạn huấn luyện
            prob_ai = prediction_prob[0]         # Xác suất ảnh là AI
            prob_real = prediction_prob[1]       # Xác suất ảnh là Thực tế
            
            # IN KẾT QUẢ
            st.subheader("💡 Kết quả phân tích:")
            
            # ========== BƯỚC MỚI: In chi tiết xác suất ==========
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🤖 AI-Generated", f"{prob_ai * 100:.2f}%")
            with col2:
                st.metric("📸 Real", f"{prob_real * 100:.2f}%")
            
            # Tính confidence gap
            confidence_gap = abs(prob_ai - prob_real) * 100
            st.info(f"📊 Margin (Độ chênh lệch): {confidence_gap:.2f}% - {'✅ Cao' if confidence_gap > 20 else '⚠️ Thấp (Model không chắc)'}")
            
            # So sánh xác suất thực tế, bạn có thể so sánh trực tiếp prob_real >= prob_ai cho an toàn
            if prob_real >= prob_ai:
                st.success(f"✅ HỆ THỐNG NHẬN ĐỊNH: Đây là **Ảnh Thực tế (Real)**")
                if confidence_gap < 20:
                    st.warning("⚠️ Cảnh báo: Model không chắc chắn. Nếu ảnh có dấu hiệu AI, hãy xem xét thêm.")
            else:
                st.error(f"🤖 HỆ THỐNG NHẬN ĐỊNH: Đây là **Ảnh do AI tạo sinh**")
                if confidence_gap < 20:
                    st.warning("⚠️ Cảnh báo: Model không chắc chắn. Cần xác nhận lại.")
            
            # ========== DEBUG MODE: In thông tin xử lý ==========
            with st.expander("🔧 Chi tiết xử lý ảnh (DEBUG)"):
                st.write(f"- **Kích thước gốc**: {uploaded_file.size} bytes")
                st.write(f"- **Kích thước sau resize**: 224x224 pixels")
                st.write(f"- **Mode ảnh**: RGB")
                st.write(f"- **Preprocessing**: MobileNetV2 (normalize [-1, 1])")
                st.write(f"- **Prediction output**: [{prob_ai:.6f}, {prob_real:.6f}]")
                st.write(f"- **Model architecture**: MobileNetV2 + Transfer Learning")