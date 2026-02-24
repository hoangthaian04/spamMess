# import pandas as pd
# import joblib
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.naive_bayes import MultinomialNB

# # Đọc dữ liệu từ file bạn vừa clone về
# df = pd.read_csv(r'D:\Slide 28 tech\Kì 2 năm 4\Chuyên đề HTTT\spam_data.csv')
# X, y = df['text'].astype(str), df['label']

# # Đóng gói bộ chuyển đổi văn bản và thuật toán vào Pipeline
# model1 = Pipeline([('tfidf', TfidfVectorizer()), ('svc', SVC(kernel='linear'))])
# model2 = Pipeline([('tfidf', TfidfVectorizer()), ('nb', MultinomialNB())])

# print("--- Đang huấn luyện mô hình... ---")
# model1.fit(X, y)
# model2.fit(X, y)

# # Lưu vào thư mục Spam để views.py có thể tải lên
# os.makedirs('Spam', exist_ok=True)
# joblib.dump(model1, 'Spam/mySVCModel1.pkl')
# joblib.dump(model2, 'Spam/myModel.pkl')
# print("--- Đã hoàn thiện 2 file mô hình! ---")




import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# 1. Cấu hình đường dẫn
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = os.path.join(BASE_DIR, 'spam_data.csv')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'Spam')

def train_and_evaluate():
    # Kiểm tra file dữ liệu
    if not os.path.exists(DATA_PATH):
        print(f"Lỗi: Không tìm thấy file {DATA_PATH}. Hãy chạy setup_data.py trước!")
        return

    # 2. Nạp dữ liệu gộp (Anh + Việt)
    print("--- Đang nạp dữ liệu song ngữ... ---")
    df = pd.read_csv(DATA_PATH)
    X = df['text'].astype(str)
    y = df['label']
    print(f"Tổng mẫu: {len(df)} (Ham: {len(df[y==0])}, Spam: {len(df[y==1])})")

    # 3. Chia dữ liệu thành 80% để học và 20% để kiểm tra thực tế
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Huấn luyện mô hình 1: Naive Bayes (Algo-2)
    print("\n--- Đang huấn luyện Algo-2 (Naive Bayes)... ---")
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('nb', MultinomialNB())
    ])
    nb_pipeline.fit(X_train, y_train)
    
    # Đánh giá Algo-2
    y_pred_nb = nb_pipeline.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # 5. Huấn luyện mô hình 2: SVC (Algo-1)
    print("--- Đang huấn luyện Algo-1 (SVC - Có thể mất 1-2 phút)... ---")
    svc_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svc', SVC(kernel='linear', C=1.0))
    ])
    svc_pipeline.fit(X_train, y_train)
    
    # Đánh giá Algo-1
    y_pred_svc = svc_pipeline.predict(X_test)
    acc_svc = accuracy_score(y_test, y_pred_svc)

    # 6. Hiển thị báo cáo kết quả
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ HIỆU SUẤT THỰC TẾ")
    print("="*50)
    print(f"Thuật toán Algo-2 (Naive Bayes) đạt độ chính xác: {acc_nb*100:.2f}%")
    print(classification_report(y_test, y_pred_nb, target_names=['Ham', 'Spam']))
    
    print("-" * 30)
    print(f"Thuật toán Algo-1 (SVC) đạt độ chính xác: {acc_svc*100:.2f}%")
    print(classification_report(y_test, y_pred_svc, target_names=['Ham', 'Spam']))
    print("="*50)

    # 7. Lưu các mô hình vào thư mục Spam
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump(nb_pipeline, os.path.join(MODEL_SAVE_DIR, 'myModel.pkl'))
    joblib.dump(svc_pipeline, os.path.join(MODEL_SAVE_DIR, 'mySVCModel1.pkl'))
    print("\n--- Đã cập nhật 2 file mô hình thành công! ---")

if __name__ == "__main__":
    train_and_evaluate()







    # cách chạy code 
    # python setup_data.py
    # python train_system.py
    # python manage.py runserver