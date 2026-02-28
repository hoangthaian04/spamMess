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
DATA_PATH = os.path.join(BASE_DIR, 'ecommerce_reviews.csv') 
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'Spam')

def train_and_evaluate():
    # Kiểm tra file dữ liệu
    if not os.path.exists(DATA_PATH):
        print(f"Lỗi: Không tìm thấy file {DATA_PATH}. Hãy chuẩn bị file dữ liệu bình luận trước!")
        return

    # 2. Nạp dữ liệu bình luận TMĐT
    print("--- Đang nạp dữ liệu bình luận Thương mại điện tử... ---")
    df = pd.read_csv(DATA_PATH)
    
    # --- THỰC HIỆN LẤY MẪU CÂN BẰNG 50/50 ---
    n_each = 10000  # Số lượng mẫu cho mỗi loại

    # Tách riêng 2 nhóm
    df_fake = df[df['label'] == 1]
    df_genuine = df[df['label'] == 0]

    # Kiểm tra số lượng tối thiểu để tránh lỗi nếu dữ liệu không đủ mẫu mỗi loại
    n_fake = len(df_fake)
    n_genuine = len(df_genuine)
    final_n = min(n_fake, n_genuine, n_each)

    # Lấy mẫu ngẫu nhiên từ mỗi nhóm
    df_fake_sampled = df_fake.sample(final_n, random_state=42)
    df_genuine_sampled = df_genuine.sample(final_n, random_state=42)

    # Gộp lại và xáo trộn (shuffle) dữ liệu
    df = pd.concat([df_fake_sampled, df_genuine_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"-> Đã lấy cân bằng: {final_n} mẫu Thật và {final_n} mẫu Ảo (Tổng: {len(df)})")
    # ----------------------------------------

    X = df['text'].astype(str)
    y = df['label'] # 1: Bình luận ảo/rác, 0: Bình luận thật
    print(f"Tổng mẫu: {len(df)} (Thật: {len(df[y==0])}, Ảo: {len(df[y==1])})")

    # 3. Chia dữ liệu 80/20 (giữ nguyên tỉ lệ nhãn bằng stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Cấu hình TF-IDF tối ưu cho Bình luận ảo
    tfidf_optimized = TfidfVectorizer(
        stop_words=None,      # Giữ lại toàn bộ từ để bắt cấu trúc câu "công nghiệp"
        ngram_range=(1, 3),   
        max_df=0.9,           
        min_df=1,             
        sublinear_tf=True,
        token_pattern=r"\b\w\w+\b|[!$]" # Giữ lại dấu chấm than và dấu $
    )

    # 5. Huấn luyện Algo-2: Naive Bayes
    print("\n--- Đang huấn luyện Algo-2 (Naive Bayes)... ---")
    nb_pipeline = Pipeline([
        ('tfidf', tfidf_optimized),
        ('nb', MultinomialNB(alpha=0.01))
    ])
    nb_pipeline.fit(X_train, y_train)
    
    y_pred_nb = nb_pipeline.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # 6. Huấn luyện Algo-1: SVC
    print("--- Đang huấn luyện Algo-1 (SVC - Cấu hình Balanced)... ---")
    svc_pipeline = Pipeline([
        ('tfidf', tfidf_optimized),
        ('svc', SVC(kernel='linear', C=1.0, probability=False))
    ])
    svc_pipeline.fit(X_train, y_train)
    
    y_pred_svc = svc_pipeline.predict(X_test)
    acc_svc = accuracy_score(y_test, y_pred_svc)

    # 7. Hiển thị báo cáo kết quả
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ HỆ THỐNG PHÁT HIỆN BÌNH LUẬN ẢO")
    print("="*50)
    print(f"Thuật toán Algo-2 (Naive Bayes) Accuracy: {acc_nb*100:.2f}%")
    print(classification_report(y_test, y_pred_nb, target_names=['Thật', 'Ảo']))
    
    print("-" * 30)
    print(f"Thuật toán Algo-1 (SVC) Accuracy: {acc_svc*100:.2f}%")
    print(classification_report(y_test, y_pred_svc, target_names=['Thật', 'Ảo']))
    print("="*50)

    # 8. Lưu mô hình
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    joblib.dump(nb_pipeline, os.path.join(MODEL_SAVE_DIR, 'myModel.pkl'))
    joblib.dump(svc_pipeline, os.path.join(MODEL_SAVE_DIR, 'mySVCModel1.pkl'))
    print("\n--- Hệ thống đã sẵn sàng cho bài toán TMĐT! ---")

if __name__ == "__main__":
    train_and_evaluate()







    # cách chạy code 
    # python setup_data.py
    # python train_system.py
    # python manage.py runserver