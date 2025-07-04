import streamlit as st
import os
import torch

from src.chatbot import LLMClient, MemoryRetrival, GenerateData
import datetime
import json
import os



def create_json_file_if_not_exists(path: str, initial_data: dict = None):
    """
    Tạo file JSON nếu chưa tồn tại.

    Args:
        path (str): Đường dẫn tới file JSON.
        initial_data (dict): Dữ liệu khởi tạo nếu cần. Mặc định là dict rỗng.
    """
    check = False
    if not os.path.exists(path):
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)
        check = True

        # Ghi file JSON rỗng hoặc có nội dung khởi tạo
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(initial_data or {}, f, ensure_ascii=False, indent=4)
        print(f"✅ Created file: {path}")
    else:
        print(f"✅ File already exists: {path}")
    return check



model_path = os.path.abspath('models/vinallama-7b-chat_q5_0.gguf')

get_data = GenerateData("intfloat/multilingual-e5-base", 'data_pdf')


# get_data.build_faiss_index()
"""
Để sử dụng chatbot vui lòng truyền câu hỏi vào query
Điền tên để chatbot ghi nhớ thông tin người dùng
Nếu đây là lần đầu sử dụng chatbot thì truyền firstTime = True
Nếu không thì truyền firstTime = False
Nếu muốn tóm tắt lại thông tin người dùng thì truyền summary = True
Nếu muốn quên thông tin người dùng thì truyền apply_forget = True
"""
def load_llm_client(user_name: str, model_path: str):
    return LLMClient(user_name, model_path)

def load_memory_retrival(user_name: str):
    return MemoryRetrival(
    "intfloat/multilingual-e5-base",  # ✅ Không có tên tham số
    5,
    200,
    user_name,
    'data/test_json.json'
    )

def test_llm_conversation(firstTime = False, query = '', user_name ='', summary = False, apply_forget = False):


  if create_json_file_if_not_exists("data/test_json.json"):
    print("Dữ liệu chưa có")
    print("Vui lòng đợi hệ thống khởi tạo")
    return


  llm_client = load_llm_client(user_name, model_path)
  cur_date = datetime.datetime.now().strftime("%Y-%m-%d")
  memory_retrival = load_memory_retrival(user_name)

  response = ''

  # Dùng cơ chế tóm tắt lại thông tin người dùng sau khi tóm tắt lại thông tin người dùng có thể refresh lại index
  if summary:
    print('Tóm tắt')
    llm_client.summary_user()
    memory_retrival.init_memory_index('data/test_json.json', 'index_storage', user_name, cur_date)
    return
  # Dùng cơ chế quên thông tin người dùng sau khi tóm tắt lại thông tin người dùng có thể refresh lại index
  if apply_forget:
    llm_client.apply_forget()
    return

  # Lần đầu chat với chatbot
  if firstTime:
    # memory_loader = MemoryForgetLoader('data/test_json.json')

    print("HELLO")
    data = get_data.query(query)

    input = [
        'data',
        'query'
    ]
    print(data)
    context = """
    Bạn là một trợ lý AI thông minh và thân thiện.

    Thông tin liên quan để trả lời câu hỏi: {data}

    Câu hỏi từ người dùng:
    {query}

    Dựa trên thông tin và dữ kiện ở trên, hãy đưa ra câu trả lời phù hợp.
    """
    # Nhận phản hồi với đây là lần đầu tiên sử sụng chatbot
    template = llm_client.create_template(input, context)
    llm_client.build_model_with_template(template)

    response = llm_client.generate_output(query, data, personal_info='')
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    # Khởi tạo một index (tương đương) 1 vector db dùng để lưu trữ file pdf
    memory_retrival.init_memory_index('data/test_json.json', 'index_storage', user_name, cur_date)
    print("DEBUGGING")


  else:
    data = get_data.query(query)

    index = memory_retrival.load_memory_index(f'index_storage/{user_name}')

    personal_info = memory_retrival.search_memory(query, index, cur_date)
    input = [
        'personal_info',
        'data',
        'query'
    ]

    context = """
    Bạn là một trợ lý AI thông minh và thân thiện.

    Thông tin cá nhân người dùng: {personal_info}
    Thông tin liên quan để trả lời câu hỏi: {data}

    Câu hỏi từ người dùng:
    {query}

    Dựa trên thông tin cá nhân và dữ kiện ở trên, hãy đưa ra câu trả lời phù hợp, đồng cảm và hữu ích. Nếu thông tin người không hữu ích hãy bỏ qua.
    """
    template = llm_client.create_template(input, context)
    llm_client.build_model_with_template(template)
    response = llm_client.generate_output(query, data, personal_info)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    # Refresh lại index mỗi lần có dữ liệu về đoạn chat mới được thêm vào (có thể tiết kiệm chi phí thì refesh sau một khoảng thời gian nào đó)
    memory_retrival.init_memory_index('data/test_json.json', 'index_storage', user_name, cur_date)




  return response if response else 'Không có câu trả lời'




if __name__ == '__main__':
   
    st.title("PDF RAG CHATBOT")

    st.markdown("""
    **Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**
    *Cách sử dụng đơn giản:**
    1. **Upload PDF** Chọn file PDF từ máy tính và nhấn "Xử lý PDF"
    2. **Đặt câu hỏi** Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức
    """)

    uploaded_file = st.file_uploader("Upload file PDF", type="pdf")

    if uploaded_file is not None:
        # Tạo thư mục lưu file
        save_dir = os.path.join(os.getcwd(), "data_pdf")
        os.makedirs(save_dir, exist_ok=True)

        # Tạo đường dẫn đầy đủ để lưu
        file_path = os.path.join(save_dir, uploaded_file.name)

        # Lưu file nhị phân
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Đã lưu file vào: {file_path}")



    handle_pdf = st.button("Xử lý PDF")
    if handle_pdf:
        get_data.build_faiss_index()

    with st.form("user_input_form"):
        user_name = st.text_input("👤 Nhập tên của bạn:", placeholder="VD: Lê Khoa")
        query = st.text_area("💬 Đặt câu hỏi:", placeholder="Tôi nên học gì để giỏi AI?")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            firstTime = st.checkbox("Lần đầu sử dụng", value=False)
        with col2:
            summary = st.checkbox("Tóm tắt thông tin", value=False)
        with col3:
            apply_forget = st.checkbox("Quên thông tin", value=False)

        submitted = st.form_submit_button("🚀 Gửi câu hỏi")

    if submitted:
        if user_name.strip() == "" or query.strip() == "":
            st.warning("⚠️ Vui lòng nhập đầy đủ tên và câu hỏi.")
        else:
            with st.spinner("💡 Đang suy nghĩ..."):
                try:
                    answer = test_llm_conversation(
                        firstTime=firstTime,
                        query=query,
                        user_name=user_name,
                        summary=summary,
                        apply_forget=apply_forget
                    )
                    st.success("✅ Trợ lý trả lời:")
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra: {e}")