# Simple_RAG_Assistant
This is an AI assistant for simple RAG based on Ollama and PyQt. 
# 电力市场知识助手
该项目实现了一个基于 PyQt 和 Ollama 的电力市场助手，并结合了一个用于检索-增强生成（RAG）的知识库，以回答与电力市场相关的问题。RAG 利用 SentenceTransformer 模型和 FAISS 向量索引检索相关文档。
## 项目简介
- **知识检索**  
  利用预先加载的知识库（文档文件，每个知识点以空行分隔），通过 SentenceTransformer 对文本进行编码，并构建 FAISS 向量索引。在用户提问时，从知识库中检索最相近的内容作为上下文，再调用 Ollama API 生成最终回答。
- **图形界面**  
  使用 PyQt 实现聊天窗口，支持自动滚动、消息气泡、头像展示等功能。
## 环境与依赖
在运行代码之前，请确保你的 Python 环境已经安装以下依赖包：
- **PyQt5**
- **requests**
- **sentence_transformers**
- **faiss-cpu** （或根据需要安装 faiss-gpu）
- **numpy**
你可以使用 pip 命令一键安装所有依赖（建议使用虚拟环境）：
```bash
pip install PyQt5 requests sentence_transformers faiss-cpu numpy
```
此外，项目中部分示例代码还使用了 `transformers` 库，如需使用请安装：
```bash
pip install transformers torch
```
## 项目文件结构
以下是项目的主要文件和目录结构示例：
```
.
├README.md
├LICENSE
├LLMassistants.py   #主要的逻辑代码
└── basis
    ├── Background.png     # 应用背景图片（通过样式表加载）
    ├── mainwin.ui         # 主界面 UI 文件（通过 Qt Designer 设计）
    ├── AI Avart.png       # AI助手头像
    ├── User Avatar.png    # 用户头像
    ├── Software Icon.png  # 软件图标
    ├── documents.txt      # 知识库文档，每个知识点之间用空行分隔
    ├── Cover.png          # 封面内容示例
    ├── Chat UI.png        # 对话内容示例
```

> **注意**：  
> 如果使用 PyInstaller 将 Python 脚本转换为 .exe 文件，生成的 **build** 文件夹和 **.spec** 文件在打包后可以删除，最终分发时请保留 **dist** 文件夹下的内容。

## 运行前的准备
在运行项目代码之前，请注意以下几点：
1. **启动 Ollama 服务**  
   项目中调用了 Ollama API 生成回答，因此请先在 CMD 中启动 Ollama 服务：
   ```bash
   ollama serve
   ```
   等待服务启动后，再运行项目代码。
2. **工作目录与资源路径**  
   - 确保项目中所有资源文件（如背景图片、头像图片、文档文件）与代码文件在正确的相对路径下。  
   - 如果遇到资源加载问题，可以检查当前工作目录（使用 `os.getcwd()`）并调整路径设置。
3. **配置 API 参数**  
   在代码中，根据实际情况修改 `OLLAMA_URL` 和 `MODEL_NAME`，确保指向正确的 Ollama API 地址和使用合适的模型。
## 运行项目
在项目根目录下打开 CMD，依次执行以下步骤：
1. **启动 Ollama 服务**  
   ```bash
   ollama serve
   ```
2. **运行主程序**  
   例如，使用 Python 运行主程序：
   ```bash
   python LLMassistants.py
   ```
   或者如果已转换为 .exe 文件，则直接运行生成的可执行文件。
项目启动后，将弹出聊天窗口，你可以在其中输入问题，系统将自动从知识库中检索相关内容，并调用 API 返回回答。
## 调试与优化
- **检索阈值**  
  如果检索到的内容与提问不匹配，可以考虑设置相似度或距离阈值，过滤掉相关性不足的检索结果。
- **界面样式**  
  可通过修改样式表（setStyleSheet）来调整聊天窗口、消息气泡、按钮等控件的外观。
- **依赖包版本**  
  建议使用最新稳定版本的依赖包，如遇兼容性问题，可尝试指定特定版本。
## 许可证
MIT
---
欢迎各位开发者提出问题和改进建议，共同完善这个项目！
