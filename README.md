# OCR长图处理器 - 模块化版本

一个专门用于处理微信聊天长图截图的OCR工具，支持智能文本识别、聊天消息结构化分析，并集成LLM对话功能。

## 项目重构说明

原始的 `refactored_ocr_processor.py` 文件已被拆分为模块化结构，提高了代码的可维护性和可扩展性。

## 新的项目结构

```
ocr_long_picture/
├── main_modular.py              # 新的主入口文件（集成LLM交互）
├── LLM_run.py                   # LLM交互模块（Ollama API）
├── refactored_ocr_processor.py  # 原始文件（保留）
├── config/                      # 配置文件
│   └── default_rapidocr.yaml    # RapidOCR配置
├── images/                      # 测试图片目录
└── ocr_long_picture/            # 模块化包
    ├── __init__.py
    ├── main.py                  # 包内主入口
    ├── core/                    # 核心处理器
    │   ├── __init__.py
    │   └── main_processor.py    # 主处理器类
    ├── processors/              # 处理器模块
    │   ├── __init__.py
    │   ├── image_processor.py   # 图像切分
    │   ├── ocr_processor.py     # OCR识别
    │   ├── avatar_detector.py   # 头像检测
    │   ├── data_deduplicator.py # 数据去重
    │   └── content_marker.py    # 内容标记
    ├── analyzers/               # 分析器模块
    │   ├── __init__.py
    │   └── chat_analyzer.py     # 聊天分析
    ├── exporters/               # 导出器模块
    │   ├── __init__.py
    │   └── result_exporter.py   # 结果导出
    ├── config/                  # 配置模块
    │   ├── __init__.py
    │   └── settings.py          # 配置常量
    └── utils/                   # 工具模块
        ├── __init__.py
        └── common.py            # 通用工具函数
```

## 主要功能

- **OCR识别**: 支持微信聊天长图的文字识别和结构化处理
- **智能分析**: 自动识别聊天消息的时间、昵称、内容等信息
- **数据统计**: 提供详细的聊天数据统计信息
- **LLM集成**: 集成本地Ollama API，支持基于聊天内容的智能问答
- **模块化设计**: 清晰的模块结构，易于维护和扩展

## 模块功能说明

### 核心模块 (core/)
- `main_processor.py`: 主处理器类，协调各个模块完成整个OCR流程

### LLM交互模块
- `LLM_run.py`: 集成本地Ollama API，实现聊天内容的智能问答功能

### 处理器模块 (processors/)
- `image_processor.py`: 负责长图切分和基础图像处理
- `ocr_processor.py`: 负责OCR文字识别
- `avatar_detector.py`: 负责头像检测和x_croped计算
- `data_deduplicator.py`: 负责OCR结果和头像位置去重
- `content_marker.py`: 负责内容标记（时间、昵称、内容）和颜色检测

### 分析器模块 (analyzers/)
- `chat_analyzer.py`: 负责将标记后的文本组织成结构化聊天消息

### 导出器模块 (exporters/)
- `result_exporter.py`: 负责结果导出为JSON格式

### 配置模块 (config/)
- `settings.py`: 存储所有配置常量（阈值、模式、颜色范围等）

### 工具模块 (utils/)
- `common.py`: 通用工具函数（IoU计算、坐标转换等）

## 环境要求

- Python 3.7+
- 本地Ollama服务（用于LLM交互）
- qwen3:8b模型（或其他兼容模型）

## 安装与配置

1. 安装依赖包：
```bash
pip install requests rapidocr-onnxruntime opencv-python pillow numpy
```

2. 安装并启动Ollama服务：
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载qwen3:8b模型
ollama pull qwen3:8b

# 启动Ollama服务
ollama serve
```

3. 准备测试图片：
   - 将微信聊天长图放入 `images/` 目录
   - 修改 `main_modular.py` 中的图片路径

## 使用方法

### 方法1：直接运行模块化版本（推荐）
```bash
python main_modular.py
```

运行后会：
1. 自动处理指定的聊天长图
2. 显示OCR识别和分析结果统计
3. 提示输入问题进行LLM交互（可选）

### 方法2：作为包使用
```python
from ocr_long_picture import RefactoredLongImageOCR

processor = RefactoredLongImageOCR()
result = processor.process_long_image("path/to/image.png")
```

### 方法3：单独使用LLM交互功能
```python
from LLM_run import process_with_llm
import json

# 加载已处理的聊天数据
with open("output_json/modified_chat.json", "r", encoding="utf-8") as f:
    chat_data = json.load(f)

# 与LLM交互
response = process_with_llm("请总结这个聊天的主要内容", chat_data["messages"])
print(response)
```

### 方法4：使用单个模块
```python
from ocr_long_picture.processors import ImageProcessor
from ocr_long_picture.processors import OCRProcessor

image_proc = ImageProcessor()
ocr_proc = OCRProcessor()
```

## 输出说明

处理完成后会生成：
- `output_images/`: 切分后的图片片段
- `output_json/modified_chat.json`: 结构化的聊天数据

## LLM交互功能

集成的LLM助手可以帮助您：
- 分析聊天内容和情感倾向
- 总结对话要点和关键信息
- 回答关于聊天记录的具体问题
- 查找特定人员的发言内容
- 分析讨论话题和参与者

## 模块化优势

1. **单一职责**: 每个模块功能独立，职责明确
2. **便于测试**: 可以单独测试各个模块
3. **易于维护**: 修改某个功能只需改对应模块
4. **代码复用**: 其他项目可以复用单个模块
5. **清晰结构**: 新人更容易理解项目架构
6. **配置集中**: 所有配置参数集中管理
7. **扩展性强**: 可以轻松添加新的处理器或分析器
8. **智能交互**: 集成LLM提供基于内容的智能问答

## 兼容性

- 原始的 `refactored_ocr_processor.py` 文件保留，确保向后兼容
- 所有功能和输出结果与原版本完全一致
- 可以无缝切换使用模块化版本

## 开发建议

1. 新功能开发请在对应模块目录下添加
2. 配置参数请添加到 `config/settings.py`
3. 通用工具函数请添加到 `utils/common.py`
4. 遵循现有的代码风格和注释规范