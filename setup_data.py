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
import zipfile
import io
import os
import kagglehub
from pathlib import Path

def get_or_clone_dataset():
    # 1. Thiết lập đường dẫn cùng cấp trong thư mục dự án
    base_dir = Path(__file__).resolve().parent
    en_file_path = os.path.join(base_dir, 'spam_data.csv')
    vi_file_path = os.path.join(base_dir, 'spamDataVN.csv')

    print("--- Bắt đầu quy trình thu thập dữ liệu tách biệt... ---")

    # --- PHẦN 1: THU THẬP & GHI DỮ LIỆU TIẾNG ANH (UCI) ---
    if not os.path.exists(en_file_path):
        try:
            print("Đang tải dữ liệu Tiếng Anh từ UCI...")
            url_uci = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
            r = requests.get(url_uci, timeout=10)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df_en = pd.read_csv(z.open('SMSSpamCollection'), sep='\t', names=['label', 'text'])
            df_en['label'] = df_en['label'].map({'spam': 1, 'ham': 0})
            
            # Ghi vào file spam_data.csv
            df_en.to_csv(en_file_path, index=False, encoding='utf-8')
            print(f"-> Đã ghi {len(df_en)} mẫu Tiếng Anh vào {en_file_path}")
        except Exception as e:
            print(f"Lỗi khi tải UCI: {e}")
    else:
        print(f"--- File Tiếng Anh '{en_file_path}' đã tồn tại. ---")

    # --- PHẦN 2: THU THẬP & GHI DỮ LIỆU TIẾNG VIỆT (KAGGLE) ---
    if not os.path.exists(vi_file_path):
        try:
            print("Đang tải dữ liệu Tiếng Việt từ Kaggle...")
            path_vi = kagglehub.dataset_download("victorhoward2/vietnamese-spam-post-in-social-network")
            files = [f for f in os.listdir(path_vi) if f.endswith('.csv')]
            
            if files:
                df_raw_vi = pd.read_csv(os.path.join(path_vi, files[0]))
                
                # Xử lý tên cột linh hoạt dựa trên file Kaggle
                df_vi = pd.DataFrame()
                df_vi['text'] = df_raw_vi['content'] if 'content' in df_raw_vi.columns else df_raw_vi.iloc[:, 1]
                
                # Chuẩn hóa nhãn để không bị mất dữ liệu khi mapping
                label_col = 'category' if 'category' in df_raw_vi.columns else df_raw_vi.columns[0]
                df_vi['label'] = df_raw_vi[label_col].astype(str).str.lower().map({
                    'spam': 1, '1': 1, '1.0': 1,
                    'ham': 0, '0': 0, '0.0': 0
                })
                
                # Ghi vào file spamDataVN.csv với encoding cho Tiếng Việt
                df_vi.dropna().to_csv(vi_file_path, index=False, encoding='utf-8-sig')
                print(f"-> Đã ghi {len(df_vi)} mẫu Tiếng Việt vào {vi_file_path}")
            else:
                print("Không tìm thấy file CSV trong Dataset Kaggle.")
        except Exception as e:
            print(f"Lỗi khi tải Kaggle: {e}")
    else:
        print(f"--- File Tiếng Việt '{vi_file_path}' đã tồn tại. ---")

    # --- PHẦN 3: ĐỌC VÀ XỬ LÝ GỘP DỮ LIỆU ---
    print("\n--- Tiến hành đọc và xử lý gộp 2 file dữ liệu... ---")
    df_en_final = pd.read_csv(en_file_path)
    df_vi_final = pd.read_csv(vi_file_path)

    df_combined = pd.concat([df_en_final, df_vi_final], ignore_index=True).dropna()
    
    print(f"Tổng mẫu sau khi gộp: {len(df_combined)}")
    print("Dữ liệu sẵn sàng cho việc huấn luyện.")
    return df_combined

if __name__ == "__main__":
    get_or_clone_dataset()