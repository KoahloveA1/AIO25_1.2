
# 📄 PDF RAG Chatbot

Ứng dụng AI hỗ trợ hỏi đáp nội dung từ tài liệu PDF bằng tiếng Việt. Sử dụng mô hình ngôn ngữ `VinaLLaMA` tích hợp cơ chế ghi nhớ dài hạn và tìm kiếm ngữ nghĩa thông minh.

---

## 🚀 Cách cài đặt môi trường

### ✅ Bước 1: Cài `llama-cpp-python` theo hệ thống của bạn

- **Nếu có GPU (NVIDIA CUDA):**

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

- **Nếu không có GPU** (không khuyến khích vì rất chậm):

```bash
pip install llama-cpp-python
```

---

### ✅ Bước 2: Cài các thư viện phụ thuộc

```bash
pip install -r requirement.txt
```

---

### ✅ Bước 3: Tải mô hình `vinallama-7b-chat_q5_0.gguf`

- Tải model tại: [🔗 Link tải VinaLLaMA 7B Chat Q5_0 (GGUF)](https://huggingface.co/vilm/vinallama-7b-chat)  

- Sau đó, lưu file model vào thư mục:

```bash
models/vinallama-7b-chat_q5_0.gguf
```

---

### ✅ Bước 4: Tạo môi trường ảo (nếu chưa có)

```bash
python -m venv venv
source venv/bin/activate      # Trên macOS/Linux
venv\Scripts\activate         # Trên Windows
```

---

### ✅ Bước 5: Chạy ứng dụng

```bash
streamlit run main.py
```

---

## 🧠 Tính năng nổi bật

- 🧾 **Truy vấn theo nội dung PDF** bằng tiếng Việt
- 🧠 **Ghi nhớ ngữ cảnh người dùng** để trả lời tự nhiên hơn
- 🧹 **Cơ chế quên & tóm tắt thông tin** người dùng để quản lý bộ nhớ
- 💡 Tích hợp FAISS để tìm kiếm thông tin hiệu quả
- 🤖 Sử dụng mô hình LLaMA chạy local — không cần internet

---

## 📂 Cấu trúc thư mục

```
├── data/                  # Chứa dữ liệu JSON, FAISS index
│   ├── test_json.json
│   ├── data_pdf/
│   ├── faiss_index/
│   └── index_storage/
├── models/                # Chứa file model GGUF
│   └── vinallama-7b-chat_q5_0.gguf
├── src/                   # Chứa mã nguồn chatbot
│   ├── chatbot.py
│   └── ...
├── main.py                # Entry point để chạy app
├── app.py                 # Có thể chứa API phụ hoặc module riêng
├── requirement.txt
└── README.md
```

---


## 🧑‍💻 Tác giả & Đóng góp

- 👤 Lê Khoa
- 📬 Liên hệ: [email@example.com](khoale11.work@gmail.com)

---
