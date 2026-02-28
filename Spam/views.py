# import pandas as pd
# import joblib
# import os
# from django.shortcuts import render
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline

# # Đường dẫn tới thư mục chứa mô hình
# MODEL_PATH = os.path.join('Spam', 'myModel.pkl')

# def index(request):
#     """Hiển thị giao diện chính của ứng dụng"""
#     return render(request, 'index.html')

# def checkSpam(request):
#     """Xử lý dự đoán tin nhắn từ form giao diện"""
#     if request.method == 'POST':
#         # 1. Lấy tin nhắn từ người dùng nhập vào ô 'message' trong form
#         message = request.POST.get('message', '')
        
#         if not message:
#             return render(request, 'index.html', {'error': 'Vui lòng nhập nội dung tin nhắn!'})

#         # 2. Kiểm tra xem file mô hình có tồn tại không
#         if not os.path.exists(MODEL_PATH):
#             return render(request, 'index.html', {'error': 'Hệ thống chưa được huấn luyện. Vui lòng chạy train_and_save_models() trước!'})

#         # 3. Tải mô hình đã lưu (bao gồm cả TfidfVectorizer và thuật toán)
#         model = joblib.load(MODEL_PATH)

#         # 4. Thực hiện dự đoán
#         # Kết quả là một mảng, ví dụ [1] cho Spam và [0] cho Ham
#         prediction = model.predict([message])[0]

#         # 5. Trả kết quả về giao diện
#         result = "Đây là TIN NHẮN RÁC (Spam)" if prediction == 1 else "Đây là TIN NHẮN THÔNG THƯỜNG (Ham)"
        
#         return render(request, 'index.html', {
#             'message': message,
#             'result': result,
#             'prediction': prediction
#         })

#     return render(request, 'index.html')

# # Giữ lại hàm huấn luyện của bạn để có thể gọi lại khi cần
# def train_and_save_models():
#     if not os.path.exists('spam_data.csv'):
#         print("Lỗi: Không tìm thấy file dữ liệu để huấn luyện!")
#         return

#     df = pd.read_csv('spam_data.csv')
#     X = df['text'].astype(str)
#     y = df['label']

#     tfidf = TfidfVectorizer(stop_words='english')
#     from sklearn.naive_bayes import MultinomialNB

#     model2 = Pipeline([('tfidf', tfidf), ('nb', MultinomialNB())])

#     print("--- Đang huấn luyện hệ thống... ---")
#     model2.fit(X, y)

#     os.makedirs('Spam', exist_ok=True)
#     joblib.dump(model2, MODEL_PATH)
#     print("--- Hệ thống đã sẵn sàng để kiểm tra tin nhắn! ---")







import os
import joblib
import re
from django.shortcuts import render
from pathlib import Path

# Cấu hình đường dẫn
MODEL_DIR = Path(__file__).resolve().parent

def preprocess_text(text):
    #Tiền xử lý văn bản Tiếng Anh cho bình luận TMĐT
    text = text.lower().strip()
    # Giữ lại các ký tự $, %, ! vì đây là dấu hiệu của bình luận quảng cáo/ảo
    text = re.sub(r'[^a-z0-9\s$%!]', '', text)
    return text

def check_heuristics(text):
    # Lọc nhanh các từ khóa rác phổ biến trong Review TMĐT
    # Các từ khóa thường thấy trong bình luận ảo/quảng cáo Tiếng Anh
    SPAM_KEYWORDS = ['free gift', 'win cash', 'click here', 'contact me', 'whatsapp']
    text_lower = text.lower()
    for word in SPAM_KEYWORDS:
        if word in text_lower:
            return 1
    return None

def index(request):
    return render(request, 'index.html')

def checkSpam(request):
    if request.method == 'POST':
        raw_message = request.POST.get('rawdata', '')
        algo_choice = request.POST.get('algo', 'Algo-1')

        if not raw_message:
            return render(request, 'index.html', {'error': 'Please enter a review to check!'})

        # 1. Tiền xử lý
        clean_text = preprocess_text(raw_message)

        # 2. Kiểm tra bộ lọc nhanh
        prediction = check_heuristics(clean_text)

        if prediction is None:
            # 3. Sử dụng Machine Learning nếu bộ lọc nhanh không khớp
            model_name = 'mySVCModel1.pkl' if algo_choice == 'Algo-1' else 'myModel.pkl'
            model_path = os.path.join(MODEL_DIR, model_name)

            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    prediction = model.predict([clean_text])[0]
                except Exception as e:
                    return render(request, 'index.html', {'error': f'Model error: {str(e)}'})
            else:
                return render(request, 'index.html', {'error': 'Model not found. Please train the system first!'})

        # 4. Trả về kết quả
        result = "FAKE / SPAM REVIEW" if prediction == 1 else "GENUINE REVIEW"
        
        return render(request, 'index.html', {
            'result': result,
            'message': raw_message,
            'prediction': prediction,
            'algo_used': algo_choice
        })

    return render(request, 'index.html')










# import joblib
# import os
# from django.shortcuts import render

# # Đường dẫn tới thư mục chứa mô hình
# MODEL_DIR = 'Spam'

# def index(request):
#     return render(request, 'index.html')

# def checkSpam(request):
#     if request.method == 'POST':
#         # 1. Lấy tin nhắn và thuật toán từ form
#         message = request.POST.get('rawdata', '') 
#         algo_choice = request.POST.get('algo', 'Algo-1')
        
#         if not message:
#             return render(request, 'index.html', {'error': 'Vui lòng nhập nội dung tin nhắn!'})

#         # 2. Xác định file model cần dùng
#         model_name = 'mySVCModel1.pkl' if algo_choice == 'Algo-1' else 'myModel.pkl'
#         model_path = os.path.join(MODEL_DIR, model_name)

#         if not os.path.exists(model_path):
#             return render(request, 'index.html', {
#                 'error': f'Mô hình {algo_choice} chưa được huấn luyện!',
#                 'message': message
#             })

#         # 3. Tải mô hình và dự đoán
#         model = joblib.load(model_path)
#         prediction = model.predict([message])[0]
        
#         result = "TIN NHẮN RÁC (Spam)" if prediction == 1 else "TIN NHẮN THÔNG THƯỜNG (Ham)"
        
#         return render(request, 'index.html', {
#             'message': message,
#             'result': result,
#             'prediction': prediction,
#             'algo_used': algo_choice # Gửi lại để HTML xử lý logic 'selected'
#         })

#     return render(request, 'index.html')











