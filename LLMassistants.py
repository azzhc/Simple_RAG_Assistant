"""
Description: This is an AI assistant for simple RAG based on Ollama and PyQt.
The knowledge base uses electricity markets as an example.
The language of communication is Chinese.
Author: ZHC
Date: 2025-3-14
"""
import sys
import requests
from PyQt5.QtWidgets import (QApplication,  QHBoxLayout, QLabel,
                             QFrame, QMainWindow,QSizePolicy)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.uic import loadUi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from numpy.linalg import norm

# User & AI Avatar path
USER_AVATAR = "./basis/User Avatar.png"
AI_AVATAR = "./basis/AI Avart.png"

# Ollama API
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2:1.5b"  # Replace with the actual model used like deepseek-r1:1.5b
HEADERS = {"Content-Type": "application/json"}

# Initial message context (contains system prompts)
messages = [{
    "role": "system",
    "content": (
        "你是电力市场百科全书，可以回答有关电力市场内容的知识。"
        "请将回答组织成流畅、连贯的段落，禁止分点回答，禁止使用任何 Markdown 或加粗符号（如**）。"
        "确保输出为正常的文本格式。"
    )
}]

# ----------------------- Document retrieval section -----------------------
class DocumentRetriever:
    """
    Document Retriever: loads documents, pre-calculates vectors and constructs the FAISS index
     (individual messages in the document file are separated by blank lines).
    """
    def __init__(self, file_path="./basis/documents.txt",#all-MiniLM-L6-v2 also ok
                 model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.file_path = file_path
        self.documents = self._read_documents()
        # load SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        # Computes the document vector and converts it to float32
        self.document_embeddings = np.array(self.model.encode(self.documents)).astype('float32')
        self.document_embeddings = self.document_embeddings / norm(self.document_embeddings, axis=1, keepdims=True)
        # Building the FAISS Index
        self.index = faiss.IndexFlatIP(self.document_embeddings.shape[1])
        self.index.add(self.document_embeddings)

    def _read_documents(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            # individual messages in the document file are separated by blank lines
            documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
            return documents
        except Exception as e:
            print(f"读取文档出错：{e}")
            return []

    def retrieve(self, query, k=1):
        """
        Retrieve the most relevant k documents according to the query statement
        and return the concatenated context string.
        """
        if not self.documents:
            return ""
        query_embedding = np.array(self.model.encode([query])).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        if distances[0][0] < 2.5:
            retrieved_docs = []
        else:
            retrieved_docs = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        context = " ".join(retrieved_docs)
        return context
# Global instance that loads documents and builds indexes at program startup
doc_retriever = DocumentRetriever(file_path="./basis/documents.txt")

# ----------------------- API request section -----------------------
def send_to_ollama_generic(input_text, input_role="user", temperature=0.7):
    """
    Generic send request function, depending on the role (user or system)
    and temperature parameters passed in.
    Appends the input context to the global messages and sends it to the Ollama API.
    """
    global messages
    messages.append({"role": input_role, "content": input_text})
    data = {
        "model": MODEL_NAME,
        "options": {
            "temperature": temperature
        },
        "stream": False,
        "messages": messages
    }
    try:
        response = requests.post(OLLAMA_URL, json=data, headers=HEADERS, timeout=60)
        if response.status_code == 200:
            result = response.json()
            output = result.get("message", {}).get("content", "无内容")
            # output = re.sub(r"(?i)think:.*", "", output)
            messages.append({"role": "system", "content": output})
            # Remove special characters from the output
            output = output.replace('*', '')
            output = output.replace('Answer:', '')
            output = output.replace('\n\n', '')
            # Truncate the context to prevent it from being too long
            if len(messages) > 50:
                messages = messages[:1] + messages[-20:]
            return output
        else:
            return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"请求失败: {e}"

# ----------------------- Asynchronous Work Threads -----------------------
class OllamaWorker(QThread):
    """Asynchronous requests to Ollama API threads (in conjunction with document retrieval)"""
    response_received = pyqtSignal(str)
    def __init__(self, message):
        super().__init__()
        self.message = message

    def run(self):
        # First use the search enhancement to generate the context
        context = doc_retriever.retrieve(self.message, k=1)
        # Constructing complete prompt messages: questions and context
        input_text = f"Question: {self.message}\nContext: {context}"
        print("发送请求：", input_text)
        response = send_to_ollama_generic(input_text, input_role="user", temperature=0.3)
        print("请求完成")
        self.response_received.emit(response)

# ----------------------- UI section -----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load the main window using the UI file (make sure mainwin.ui exists)
        loadUi("./basis/mainwin.ui", self)
        self.setWindowTitle('电力市场百科全书')
        self.setWindowIcon(QIcon('./basis/Software Icon.png'))
        self.cover.setStyleSheet("""
            background-image: url('./basis/Background.png');
            background-position: top left;
            background-repeat: no-repeat;
            background-size: 100% 100%;
        """)
        self.start.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.send.clicked.connect(self.send_message)
        self.chat_container.setSpacing(10)
    def send_message(self):
        """User sends a message"""
        user_text = self.text_input.toPlainText().strip()
        if user_text:
            self.add_message(user_text, "user")
            self.text_input.clear()
            self.worker = OllamaWorker(user_text)
            self.worker.response_received.connect(self.display_ai_response)
            self.worker.start()

    def display_ai_response(self, response):
        """Show AI Response"""
        self.add_message(response, "ai")

    def add_message(self, text, sender):
        """Add message bubbles to the interface and
         adjust the alignment according to the sender"""
        message_label = QLabel(text)
        max_width = int(self.width() * 0.7)
        message_label.setFixedWidth(max_width)
        message_label.setWordWrap(False)
        message_label.setStyleSheet("font-size: 18px; padding: 10px; border-radius: 10px;")
        font_metrics = message_label.fontMetrics()
        text_width = font_metrics.boundingRect(message_label.text()).width()+18*5# 加内边距
        bubble_width = min(text_width, max_width)
        message_label.setFixedWidth(bubble_width)
        message_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        message_label.setWordWrap(True)
        # message_label.adjustSize()
        if sender == "user":
            message_label.setStyleSheet(message_label.styleSheet() + "background-color: #A7C7E7;")
        else:
            message_label.setStyleSheet(message_label.styleSheet() + "background-color: #89D961;")
        # Avatar
        avatar_label = QLabel()
        avatar_label.setFixedSize(40, 40)
        avatar_label.setScaledContents(True)
        avatar_path = USER_AVATAR if sender == "user" else AI_AVATAR
        avatar_label.setPixmap(QPixmap(avatar_path))
        avatar_label.setStyleSheet("border-radius: 20px;")

        # Layout settings: adjust alignment order according to sender
        message_layout = QHBoxLayout()
        message_layout.setAlignment(Qt.AlignTop)
        if sender == "user":
            message_layout.addStretch()
            message_layout.addWidget(message_label)
            message_layout.addWidget(avatar_label)
        else:
            message_layout.addWidget(avatar_label)
            message_layout.addWidget(message_label)
            message_layout.addStretch()
        container = QFrame()
        container.setLayout(message_layout)
        container.setContentsMargins(5, 5, 5, 5)
        container.setMinimumHeight(message_label.sizeHint().height())
        self.chat_container.addWidget(container)
        # self.chat_container.addSpacing(5)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
