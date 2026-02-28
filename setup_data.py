# import pandas as pd
# import requests
# import zipfile
# import io
# import os
# import kagglehub

# def get_or_clone_dataset():
#     # Sử dụng raw string để tránh lỗi đường dẫn Windows
#     file_path = r'D:\Slide 28 tech\Kì 2 năm 4\Chuyên đề HTTT\spam_data.csv'
    
#     # 1. Kiểm tra nếu file đã tồn tại thì đọc luôn, không tải lại
#     if os.path.exists(file_path):
#         print(f"--- File '{file_path}' đã có trên máy. Đang đọc dữ liệu... ---")
#         return pd.read_csv(file_path)

#     print("--- Bắt đầu quy trình thu thập dữ liệu song ngữ... ---")
    
#     # --- PHẦN 1: TẢI DỮ LIỆU TIẾNG ANH (UCI) ---
#     try:
#         url_uci = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
#         r = requests.get(url_uci, timeout=10)
#         z = zipfile.ZipFile(io.BytesIO(r.content))
        
#         df_en = pd.read_csv(z.open('SMSSpamCollection'), sep='\t', names=['label', 'text'])
#         # Chuyển nhãn về dạng số: spam=1, ham=0
#         df_en['label'] = df_en['label'].map({'spam': 1, 'ham': 0})
#         print(f"-> Đã tải {len(df_en)} mẫu Tiếng Anh.")
#     except Exception as e:
#         print(f"Lỗi khi tải UCI: {e}")
#         df_en = pd.DataFrame(columns=['label', 'text'])

#     # --- PHẦN 2: TẢI DỮ LIỆU TIẾNG VIỆT (KAGGLE) ---
#     try:
#         # Tải dataset từ KaggleHub
#         path_vi = kagglehub.dataset_download("victorhoward2/vietnamese-spam-post-in-social-network")
        
#         # Tìm file csv trong thư mục vừa tải về
#         files = [f for f in os.listdir(path_vi) if f.endswith('.csv')]
#         if files:
#             df_vi_raw = pd.read_csv(os.path.join(path_vi, files[0]))
            
#             # Chuẩn hóa tên cột để khớp với df_en (ví dụ: category -> label, content -> text)
#             # Lưu ý: Tên cột có thể thay đổi tùy version dataset, hãy kiểm tra df_vi_raw.columns nếu lỗi
#             df_vi = pd.DataFrame()
#             df_vi['text'] = df_vi_raw['content'] # Giả định cột nội dung là 'content'
#             df_vi['label'] = df_vi_raw['category'].map({'spam': 1, 'ham': 0}) # Giả định cột nhãn là 'category'
            
#             print(f"-> Đã tải {len(df_vi)} mẫu Tiếng Việt từ Kaggle.")
#         else:
#             df_vi = pd.DataFrame(columns=['label', 'text'])
#     except Exception as e:
#         print(f"Lỗi khi tải Kaggle: {e}")
#         df_vi = pd.DataFrame(columns=['label', 'text'])

#     # --- PHẦN 3: GỘP VÀ LƯU TRỮ ---
#     # Kết hợp hai DataFrame lại với nhau
#     df_combined = pd.concat([df_en, df_vi], ignore_index=True).dropna()
    
#     # Lưu với utf-8-sig để hiển thị tốt tiếng Việt trong Excel và Python
#     df_combined.to_csv(file_path, index=False, encoding='utf-8-sig')
    
#     print(f"--- HOÀN TẤT ---")
#     print(f"Tổng cộng {len(df_combined)} mẫu (Anh + Việt) đã được lưu tại: {file_path}")
#     return df_combined

# if __name__ == "__main__":
#     get_or_clone_dataset()






import pandas as pd
import requests
import os
from pathlib import Path

def get_or_clone_ecommerce_dataset():
    # 1. Thiết lập đường dẫn file dữ liệu mới
    base_dir = Path(__file__).resolve().parent
    file_path = os.path.join(base_dir, 'ecommerce_reviews.csv')

    print("--- Bắt đầu quy trình thu thập dữ liệu Bình luận TMĐT ---")

    # --- PHẦN 1: TẢI DỮ LIỆU FAKE REVIEWS (AMAZON) ---
    if not os.path.exists(file_path):
        try:
            # URL bộ dữ liệu Fake Reviews từ GitHub (Dữ liệu sạch, sẵn sàng cho ML)
            url = "https://raw.githubusercontent.com/SayamAlt/Fake-Reviews-Detection/main/fake%20reviews%20dataset.csv"
            
            print("Đang tải dữ liệu Amazon Reviews (khoảng 15MB)...")
            r = requests.get(url, timeout=30)
            
            # Đọc dữ liệu từ nội dung tải về
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            
            # 2. Chuẩn hóa dữ liệu
            # chỉ lấy cột văn bản và nhãn
            df = df[['text_', 'label']]
            df.columns = ['text', 'label_raw']
            
            # 3. Chuyển đổi nhãn: 'fake' -> 1 (Spam/Ảo), 'genuine' -> 0 (Thật)
            # Chuẩn hóa về 1 và 0
            df['label'] = df['label_raw'].apply(lambda x: 1 if str(x).lower() in ['fake', 'cg'] else 0)
            
            # Xóa cột nhãn gốc và các dòng trống/trùng
            df = df[['text', 'label']].dropna().drop_duplicates()
            
            # 4. Ghi vào file ecommerce_reviews.csv
            df.to_csv(file_path, index=False, encoding='utf-8')
            print(f"-> Thành công! Đã lưu {len(df)} bình luận vào {file_path}")
            print(f"Phân bổ nhãn: \n{df['label'].value_counts()}")
            
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu TMĐT: {e}")
    else:
        print(f"--- File dữ liệu '{file_path}' đã tồn tại. ---")

    # Đọc lại để kiểm tra
    df_final = pd.read_csv(file_path)
    return df_final

if __name__ == "__main__":
    get_or_clone_ecommerce_dataset()