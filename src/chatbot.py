import copy
import json
import PyPDF2

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

import re
from typing import List, Tuple
import datetime
import os
import math
import random
import numpy as np




def clean_text(text: str) -> str:
    """
    Giữ lại cấu trúc dạng đoạn văn hoặc liệt kê.
    Xóa khoảng trắng đầu dòng, dòng trống, và chuẩn hóa dòng liên tục.
    """
    # Tách từng dòng, loại bỏ khoảng trắng đầu/cuối từng dòng
    lines = [line.strip() for line in text.strip().splitlines()]

    # Loại bỏ dòng trống
    lines = [line for line in lines if line]

    # Nếu toàn bộ chỉ là một đoạn văn không xuống dòng có chủ ý, nối lại một dòng
    if all(not re.match(r'^[-•*]|^[A-Z][a-z]+:', line) for line in lines):
        return ' '.join(lines)

    # Ngược lại: giữ xuống dòng giữa các đoạn có ý nghĩa
    return '\n'.join(lines)


# Class khởi tạo model llms
class LLMs:
    def __init__(self, model_id):
        self.__model = LlamaCpp(
            model_path=model_id,
            n_gpu_layers=5,             # ⚠️ Giới hạn an toàn cho 4GB (thử 10–16 tùy model)
            n_ctx=4096,                  # ⚠️ Giảm để tiết kiệm VRAM (4096 nếu RAM đủ)
            max_tokens=256,              # Giữ nguyên
            temperature=0.1,
            top_p=0.95,
            n_threads=8,                 # Phụ thuộc số CPU
            n_batch=64,                  # ⚠️ Giảm batch size để giảm memory peak
            use_mmap=True,
            use_mlock=True,              # Giữ model trong RAM
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

    def build_llm(self, prompt_template: PromptTemplate):
        self.__llm_chain = LLMChain(llm=self.__model, prompt=prompt_template)
        return self.__llm_chain

    def get_model(self):
        return self.__model
    def generate_output(self, input_dict: dict):
        if not hasattr(self, '_LLMs__llm_chain'):
            raise RuntimeError("LLM chain not built. Call build_llm() first.")
        raw_output = self.__llm_chain.run(input_dict)
        return clean_text(raw_output)

    


# Cơ chế quên được xây dựng hàm log
def forgetting_curve(t, S):
    """
    Calculate the retention of information at time t based on the forgetting curve.

    :param t: Time elapsed since the information was learned (in days).
    :type t: float
    :param S: Strength of the memory.
    :type S: float
    :return: Retention of information at time t.
    :rtype: float
    Memory strength is a concept used in memory models to represent the durability or stability of a memory trace in the brain.
    In the context of the forgetting curve, memory strength (denoted as 'S') is a parameter that
    influences the rate at which information is forgotten.
    The higher the memory strength, the slower the rate of forgetting,
    and the longer the information is retained.
    """
    return math.exp(-t / 5*S)    





def get_docs_with_score(docs_with_score):
    docs=[]
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i-1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, len(self.index_to_docstore_id)-i)):
                for l in [i+k, i-k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        # print(doc0.metadata)
                        # exit()
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break
                        # print(doc0)
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
        id_list = sorted(list(id_set))
        id_lists = seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append((doc, scores[0][j]))
        return docs



class MemoryForgetLoader:
    # Khởi tạo một dict tạm để lưu trũ các thông tin.
    def __init__(self, file_path):
        self.file_path = file_path
        self.memory_bank = {}

    def _get_date_difference(self, date1: str, date2: str) -> int:
        date_format = "%Y-%m-%d"
        d1 = datetime.datetime.strptime(date1, date_format)
        d2 = datetime.datetime.strptime(date2, date_format)
        return (d2 - d1).days


    def write_memories(self, out_file):
        with open(out_file, "w", encoding="utf-8") as f:
            print(f'Successfully write to {out_file}')
            json.dump(self.memory_bank, f, ensure_ascii=False, indent=4)

    def load_memories(self, memory_file):
        # print(memory_file)
        with open(memory_file, "r", encoding="utf-8") as f:
            self.memory_bank = json.load(f)
    # Hàm khởi tạo và cập nhật metadata vào db có kèm theo cơ chế  suy giảm trí nhớ.
    def update_forget_memory(self, name, cur_date):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        docs_about_user = []
        for user_name, info in data.items():
            self.memory_bank[user_name] = copy.deepcopy(info)
            if user_name != name:
                continue
            if 'history' not in info.keys():
                continue
            for date, dialog in info['history'].items():
                # Khởi tạo mảng các đoạn đối thoại sẽ bị quên của user trong date
                forget_index = []
                tmp_str = f"Đây là đoạn đối thoại vào {date} "
                for i, chat in enumerate(dialog):

                    # Khởi tạo metadata hoặc lấy metadata để trích xuất quên của chabot về đoạn hội thoại nào đó
                    memory_strength = chat.get('memory_strength', 1)
                    last_recall_date = chat.get('last_recall_date', date)
                    memory_id = chat.get('memory_id', f'{user_name}_{date}_{i}')
                    query = f"[|User|]: {chat['query']}"
                    response = f"[|AI|]: {chat['response']}"
                    tmp_str += query + response
                    metadata = {
                        'memory_strength': memory_strength,
                        'last_recall_date': last_recall_date,
                        'memory_id': memory_id
                    }
                    self.memory_bank[user_name]['history'][date][i].update(metadata)

                    diff_date = self._get_date_difference(last_recall_date, cur_date)
                    forget_probability = forgetting_curve(diff_date, memory_strength)
                    # Thử khả năng liệu có quên hay không?
                    if random.random() > forget_probability:
                        forget_index.append(i)
                    else:
                        # Lưu thông tin người dùng để tạo documents với metadata để dễ retrival
                        docs_about_user.append(Document(page_content=tmp_str,metadata=metadata))
                if len(forget_index) > 0:
                    forget_index.sort(reverse=True)
                    for idd in forget_index:
                        self.memory_bank[user_name]['history'][date].pop(idd)
                        print(f'Delete convestion of {user_name} on {date}')
                if len(self.memory_bank[user_name]['history'][date]) == 0:
                    self.memory_bank[user_name]['history'].pop(date)
                    self.memory_bank[user_name]['summary'].pop(date)

                if 'summary' in info.keys():
                    if date in self.memory_bank[user_name]['summary'].keys():
                        summary = f"Đây là tóm tắt vào ngày {date}"
                        summary += data[user_name]['summary'][date]['content']
                        memory_strength = self.memory_bank[user_name]['summary'][date].get('memory_strength',1)
                        last_recall_date = self.memory_bank[user_name]["summary"][date].get('last_recall_date',date)
                        metadata = {
                            'memory_strength':memory_strength,
                            'memory_id':f'{user_name}_{date}_summary',
                            'last_recall_date':last_recall_date,"source":f'{user_name}_{date}_summary'
                        }
                        self.memory_bank[user_name]['summary'][date].update(metadata)
                        docs_about_user.append(Document(page_content=summary,metadata=metadata))
                # if 'overall_history' in info.keys():
                #     metadata = {
                #         'overall_history' : user_name
                #     }
                #     docs_about_user.append(Document(page_content=data[user_name]['overall_history'], metadata=metadata))
                # if 'overall_personality' in info.keys():
                #     metadata = {
                #         'overall_personality' : user_name
                #     }
                #     docs_about_user.append(Document(page_content=data[user_name]['overall_personality'], metadata=metadata))
        self.write_memories(self.file_path)
        return docs_about_user

    def update_memory_when_searched(self, recalled_memos,user,cur_date):
        for recalled in recalled_memos:
            recalled_id = recalled.metadata['memory_id']
            recalled_date = recalled_id.split('_')[1]
            for i,memory in enumerate(self.memory_bank[user]['history'][recalled_date]):
                if memory['memory_id'] == recalled_id:
                    self.memory_bank[user]['history'][recalled_date][i]['memory_strength'] += 1
                    self.memory_bank[user]['history'][recalled_date][i]['last_recall_date'] = cur_date
                    break



# Class khởi tạo một instance có khả năng truy xuất bộ nhớ.
from types import MethodType
class MemoryRetrival:
    def __init__(self, embedding_name, top_k, chunk_size, user, file_path):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_name)

        self.top_k = top_k
        self.chunk_size = chunk_size
        self.user = user
        self.memory_path = file_path

    def helper_load_file(self, file_name: str, user_name: str, cur_date):
        loader = MemoryForgetLoader(file_name)
        docs = loader.update_forget_memory(user_name,cur_date)
        splitter = RecursiveCharacterTextSplitter()
        docs = splitter.split_documents(docs)
        return docs, loader

    def init_memory_index(self, file_name: str, saving_path: str, user_name: str, cur_date: str):
        # Load và chunk văn bản
        docs, self.memory_loader = self.helper_load_file(file_name, user_name, cur_date)

        if not docs:
            print("❌ Không có tài liệu nào được tạo.")
            return

        # Tạo FAISS index từ tài liệu
        vector_store = FAISS.from_documents(docs, self.embedding_model)

        # Đảm bảo thư mục tồn tại
        os.makedirs(saving_path, exist_ok=True)

        # Đặt tên file FAISS theo user_name
        file_id = user_name or f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        full_path = os.path.join(saving_path, file_id)

        # Lưu FAISS
        vector_store.save_local(full_path)
        print(f"📦 FAISS index đã được lưu tại: {full_path}")


    def load_memory_index(self, vs_path: str):
        """Tải FAISS index đã lưu từ vs_path"""
        cur_date = datetime.datetime.now().strftime("%Y-%m-%d")
        docs, self.memory_loader = self.helper_load_file(self.memory_path, self.user, cur_date)
        vector_store = FAISS.load_local(
            vs_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        print("📊 FAISS index info:")
        print(" - Total docs in index:", len(vector_store.index_to_docstore_id))
        print(" - Index dimension:", vector_store.index.d)
        print(" - Query vector dimension:", len(self.embedding_model.embed_query("So sánh esp32 và arduino")))
        # Gán hàm tìm kiếm custom nếu có (tối ưu ghép chunk)
        FAISS.similarity_search_with_score_by_vector = MethodType(similarity_search_with_score_by_vector, vector_store)

        # Gán chunk_size cho vector_store để sử dụng trong search nếu cần
        vector_store.chunk_size = self.chunk_size

        print(f"✅ Đã load memory index từ: {vs_path}")
        return vector_store


    def search_memory(self, query: str, vector_store, cur_date=''):
        print("📌 Query:", query)
        print("📌 Query embedding:", self.embedding_model.embed_query(query)[:10])  # in 10 số đầu

        # Test truy xuất
        test_docs = vector_store.similarity_search(query, k=3)
        print("📄 Top retrieved docs:")
        for i, d in enumerate(test_docs):
            print(f"{i+1}. {d.page_content[:100]}...")  # in 100 ký tự đầu mỗi doc
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        related_docs = get_docs_with_score(related_docs_with_score)

        # ✅ Tách ngày từ memory_id để sắp theo ngày + memory_id
        def extract_date(doc):
            memory_id = doc.metadata["memory_id"]
            return memory_id.split("_")[1]  # ví dụ: "2025-06-25"

        related_docs = sorted(
            related_docs,
            key=lambda x: (extract_date(x), x.metadata["memory_id"])
        )

        pre_date = ''
        date_docs = []
        dates = []
        cur_date = cur_date if cur_date else datetime.date.today().strftime("%Y-%m-%d")

        for doc in related_docs:
            date_str = extract_date(doc)
            doc.page_content = doc.page_content.replace(f'Đây là đoạn đối thoại vào {date_str}：', '').strip()

            if date_str != pre_date:
                date_docs.append(doc.page_content)
                pre_date = date_str
                dates.append(pre_date)
            else:
                date_docs[-1] += f'\n{doc.page_content}'

        self.memory_loader.update_memory_when_searched(related_docs, user=self.user, cur_date=cur_date)
        self.save_updated_memory()
        return date_docs, ', '.join(dates)

    def save_updated_memory(self):
        self.memory_loader.write_memories(self.memory_path)#.replace('.json','_forget_format.json'))



class LLMClientSimple:
    def __init__(self, llm_chain):
        self.llm_chain = llm_chain  # LLMChain từ mô hình local đã được build

    def generate_text_simple(self, prompt, prompt_num=1, language='vi'):
        """
        Sinh văn bản từ local model với prompt đầu vào.
        prompt_num không áp dụng với local LLM, chỉ giữ lại cho tương thích.
        """
        try:
            result = self.llm_chain.invoke({"text": prompt})
            # Nếu LLMChain trả về {'text': "..."} thì chỉ lấy phần nội dung
            return result.get("text", "") if isinstance(result, dict) else result
        except Exception as e:
            print(f"Lỗi khi sinh văn bản từ local model: {e}")
            return ""

# Tạo prompt cho tóm tắt

def summarize_content_prompt(content, user_name, bot_name):
    prompt = "Hãy tóm tắt nội dung cuộc hội thoại sau bằng tiếng Việt, rút ra chủ đề chính và những thông tin quan trọng:\n"
    for dialog in content:
        prompt += f"{user_name}: {dialog['query'].strip()}\n"
        prompt += f"{bot_name}: {dialog['response'].strip()}\n"
    prompt += "Tóm tắt:\n"
    return prompt

def summarize_overall_prompt(content):
    prompt = "Hãy tóm tắt ngắn gọn những sự kiện đã diễn ra dưới đây, chỉ giữ lại các thông tin quan trọng nhất:\n"
    for date, summary_dict in content.items():
        # summary_text = summary_dict.get("content", "").strip()
        prompt += f"- Ngày {date}: {summary_dict}\n"
    prompt += "Tóm tắt tổng quát:\n"
    return prompt

def summarize_personality_prompt(content, user_name, bot_name):
    prompt = f"Hãy dựa vào đoạn hội thoại sau để phân tích tính cách và cảm xúc của {user_name}, đồng thời đề xuất chiến lược phản hồi phù hợp cho {bot_name}:\n"
    for dialog in content:
        prompt += f"{user_name}: {dialog['query'].strip()}\n"
        prompt += f"{bot_name}: {dialog['response'].strip()}\n"
    prompt += f"\nTính cách, cảm xúc của {user_name} và chiến lược phản hồi của {bot_name} là:\n"
    return prompt

def summarize_overall_personality(content):
    prompt = "Dưới đây là các phân tích về tính cách và cảm xúc người dùng trong nhiều đoạn hội thoại:\n"
    for date, summary in content.items():
        prompt += f"- Ngày {date}: {summary}\n"
    prompt += "\nVui lòng tổng hợp thành một bản tóm tắt ngắn gọn về tính cách tổng thể của người dùng và cách phản hồi phù hợp nhất từ AI:\n"
    return prompt



# Tóm tắt lại các đoạn đội thoại của user_name tương ứng và lưu vào file db

def summarize_memory(memory_path, name, llm_client):
    bot_name = "AI"
    gen_prompt_num = 1
    with open(memory_path, 'r', encoding='utf8') as f:
        memory = json.load(f)

    for user_name, user_data in memory.items():
        if name is not None and user_name != name:
            continue

        print(f"Updating memory for user: {user_name}")

        history = user_data.get("history", {})
        if not history:
            continue

        user_data.setdefault("summary", {})
        user_data.setdefault("personality", {})

        for date, content in history.items():

            content_prompt = summarize_content_prompt(content, user_name, bot_name)
            personality_prompt = summarize_personality_prompt(content, user_name, bot_name)

            summary_text = llm_client.generate_text_simple(prompt=content_prompt, prompt_num=gen_prompt_num, language="vi")
            user_data["summary"][date] = {"content": ""}
            user_data["summary"][date]['content'] = summary_text

            personality_text = llm_client.generate_text_simple(prompt=personality_prompt, prompt_num=gen_prompt_num, language="vi")
            user_data["personality"][date] = personality_text

        overall_content_prompt = summarize_overall_prompt(user_data["summary"])
        overall_personality_prompt = summarize_overall_personality(user_data["personality"])
        user_data["overall_history"] = {'content': ''}
        user_data["overall_personality"] = {'content': ''}

        user_data["overall_history"]['content'] = llm_client.generate_text_simple(prompt=overall_content_prompt, prompt_num=gen_prompt_num, language="vi")
        user_data["overall_personality"]['content'] = llm_client.generate_text_simple(prompt=overall_personality_prompt, prompt_num=gen_prompt_num, language="vi")

    with open(memory_path, 'w', encoding='utf8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)
        print(f"Memory updated for {'all users' if name is None else name}")

    return memory



# Class application được dùng khi đã tích hợp đầy đủ các cơ chế
class LLMClient:
    def __init__(self, user_name=None, model_path='../models/vinallama-7b-chat_q5_0.gguf'):
        self.user_name = user_name
        self.file_path_db = os.path.join(os.getcwd(), "data/test_json.json")
        self.llm = LLMs(model_path)

    def create_template(self, data: List[str], instruction: str):
        """
        Tạo prompt_template với:
            data: là 1 list str bao gồm các dữ liệu như thông tin về  AI ghi nhớ và dữ liệu muốn AI trả lời
            instruction: Hướng dẫn chatbot trả lời theo format mong muốn
        """
        template = PromptTemplate(
            input_variables = data,
            template=instruction
        )
        return template

    def build_model_with_template(self, prompt_template):
        self.chatbot = self.llm.build_llm(prompt_template)
    # Need to update
    def summary_user(self):
        # Mẫu prompt đơn giản dùng cho tóm tắt nội dung
        default_prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Dưới đây là đoạn hội thoại giữa một con người và một trợ lý AI thông minh, am hiểu về tâm lý học.\n"
                "Người dùng: Xin chào! Hãy giúp tôi tóm tắt nội dung của đoạn hội thoại.\n"
                "AI: Tất nhiên, tôi sẽ cố gắng hết sức để hỗ trợ bạn.\n"
                "Người dùng: {text}\n"
            )
        )
        print('▶️ Bắt đầu cập nhật thông tin người dùng từ hội thoại...')
        tmp_model = self.llm.build_llm(default_prompt)
        client_simple = LLMClientSimple(tmp_model)
        summarize_memory(self.file_path_db, self.user_name, client_simple)


    def add_conservation_to_db(self, chat: dict):
        """
        Thêm một đoạn hội thoại vào database JSON hiện có (ở self.file_path_db).
        Nếu file chưa tồn tại thì tạo mới.
        Dữ liệu đầu vào phải theo format:
        {
            "2025-07-01": [  # ngày
                {
                    "query": "...",
                    "response": "...",
                    "memory_strength": ...,
                    "last_recall_date": "...",
                    "memory_id": "..."
                }
            ]
        }
        """

        # Đọc dữ liệu hiện có
        if os.path.exists(self.file_path_db):
            with open(self.file_path_db, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        # Khởi tạo user nếu chưa có
        if self.user_name not in data:
            data[self.user_name] = {
                "history": {},
                "summary": {},
                "personality": {},
                "overall_history": "",
                "overall_personality": ""
            }

        # Thêm chat vào phần history
        for date, messages in chat.items():

            if date not in data[self.user_name]["history"]:
                data[self.user_name]["history"][date] = []

            if isinstance(messages, dict):
                data[self.user_name]["history"][date].append(messages)

        # Ghi lại vào file
        with open(self.file_path_db, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def apply_forget(self):
      memory_loader = MemoryForgetLoader(self.file_path_db)
      memory_loader.update_forget_memory(self.user_name, datetime.datetime.now().strftime("%Y-%m-%d"))


    def generate_output(self, query: str, data, personal_info) -> str:
        """
        Sinh câu trả lời từ mô hình với câu hỏi `query` và ngữ cảnh `context` (nếu có),
        sau đó lưu đoạn hội thoại vào database JSON.
        """
        if not hasattr(self, "chatbot"):
            raise RuntimeError("Model chưa được khởi tạo. Hãy gọi build_model_with_template() trước.")

        # Chuẩn bị input cho LLMChain
        if personal_info == '':
            input_data = {
                        "data" : data,
                        "query" : query
                        }           
        else:
            input_data = {"personal_info" : personal_info,
                        "data" : data,
                        "query" : query
                        }

        # Sinh câu trả lời từ mô hình
        response = self.llm.generate_output(input_data)
        print("🔍 DEBUG RESPONSE RAW:", response)

        # Lấy ngày hôm nay dạng yyyy-mm-dd
        today = datetime.date.today().isoformat()

        # Tạo đoạn hội thoại để lưu
        new_entry = {
            "query": query,
            "response": response,
            "memory_strength": 1,  # có thể thay đổi theo logic AI
            "last_recall_date": today,
            "memory_id": f"{self.user_name}_{today}_{int(datetime.datetime.now().timestamp())}"
        }

        # Đóng gói thành dict để lưu vào DB
        chat_to_save = {
                today: new_entry
        }

        # Lưu vào JSON DB
        self.add_conservation_to_db(chat_to_save)

        return response






def clean_pdf_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Loại bỏ các dòng chứa số trang hoặc header/footer lặp lại
        if re.search(r'Page \d+/\d+', line):
            continue
        if "Trường Đại Học Bách Khoa" in line:
            continue
        if "Khoa Khoa Học Và Kĩ Thuật Máy Tính" in line:
            continue
        if "Assignment Software Engineering" in line:
            continue
        cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines)
class GenerateData:
    def __init__(self, embedding_name: str, folder_data: str, faiss_save_path: str = "faiss_index"):
        self.model_embedding = HuggingFaceEmbeddings(model_name=embedding_name)

        self.folder_data = folder_data
        self.faiss_save_path = faiss_save_path
        self.documents = []
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100
        )
    def load_pdf_documents(self):
        """Đọc mỗi file PDF thành 1 Document duy nhất, chỉ lưu metadata là tên file"""
        for filename in os.listdir(self.folder_data):
            if filename.lower().endswith(".pdf"):
                filepath = os.path.join(self.folder_data, filename)
                try:
                    with open(filepath, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() or ""
                    if text.strip():
                        metadata = {"source": filename}  # ✅ chỉ lưu tên file
                        self.documents.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    print(f"❌ Lỗi đọc {filename}: {e}")
        return self.documents

    def split_documents(self):
        """Tách nhỏ tài liệu thành các chunk để đưa vào FAISS"""
        if not self.documents:
            self.load_pdf_documents()
        return self.text_splitter.split_documents(self.documents)

    def build_faiss_index(self, save: bool = True):
        """Tạo FAISS index từ các document đã split"""
        docs = self.split_documents()
        self.vectorstore = FAISS.from_documents(docs, self.model_embedding)

        if save:
            self.vectorstore.save_local(self.faiss_save_path)
            print(f"✅ FAISS index đã được lưu tại: {self.faiss_save_path}")
        return self.vectorstore

    def load_faiss_index(self):
        """Load FAISS index từ thư mục đã lưu"""
        self.vectorstore = FAISS.load_local(
            self.faiss_save_path,
            self.model_embedding,
            allow_dangerous_deserialization=True
        )

        # ✅ Gán chunk_size vào vectorstore
        self.vectorstore.chunk_size = 200  # hoặc truyền từ self nếu bạn muốn linh hoạt

        # ✅ Gán lại method nếu đang dùng FAISS custom
        self.vectorstore.similarity_search_with_score_by_vector = MethodType(
            similarity_search_with_score_by_vector,
            self.vectorstore
        )

        print(f"✅ FAISS index đã được load từ: {self.faiss_save_path}")
        return self.vectorstore

    def query(self, question: str, k: int = 3):
        """Truy vấn câu hỏi vào FAISS và trả về top-k câu trả lời"""
        if not self.vectorstore:
            self.load_faiss_index()
        if not hasattr(self.vectorstore, "chunk_size"):
            self.vectorstore.chunk_size = 200
        results = self.vectorstore.similarity_search(question, k=k)
        return [clean_pdf_text(x.page_content) for x in results]