# OCR Long Picture Processor

A professional OCR tool for processing long chat screenshots, extracting messages, and analyzing conversations using advanced image processing and LLM integration.

## Features

- **Long Image Processing**: Automatically slices long screenshots into manageable pieces
- **Avatar Detection**: Identifies user avatars to determine message ownership
- **OCR Recognition**: Extracts text using RapidOCR with high accuracy
- **Content Classification**: Automatically categorizes text as nicknames, messages, or timestamps
- **LLM Integration**: Optional integration with Ollama for intelligent conversation analysis
- **Structured Output**: Exports results in JSON format for further processing

## Project Structure

```
├── main.py                      # Main entry point with CLI interface
├── refactored_ocr_processor.py  # Core OCR processing engine
├── process_avatar.py            # Avatar detection and image preprocessing
├── LLM_run.py                  # LLM integration (Ollama)
├── config/                     # Configuration files
│   └── default_rapidocr.yaml   # RapidOCR configuration
├── images/                     # Input images
├── fonts/                      # Fonts for visualization
└── requirements.txt            # Python dependencies
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Install Ollama for LLM features:
```bash
# Follow instructions at https://ollama.ai/
ollama pull qwen3:8b
```

## Usage

### Basic Usage

Process a long screenshot:
```bash
python main.py --image images/your_screenshot.png
```

### Advanced Options

```bash
# Use custom config
python main.py --image images/screenshot.png --config config/custom_rapidocr.yaml

# Skip LLM interaction
python main.py --image images/screenshot.png --no-llm
```

### Direct Python Usage

```python
from refactored_ocr_processor import RefactoredLongImageOCR

processor = RefactoredLongImageOCR()
result = processor.process_long_image("images/screenshot.png")
print(f"Processed {result['metadata']['total_messages']} messages")
```

## Output

The tool generates several output files:

- `output_json/marked_ocr_results_original.json`: OCR results with classification tags
- `output_json/structured_chat_messages.json`: Organized conversation data
- `output_images/`: Visual debugging and intermediate results

## Configuration

Edit `config/default_rapidocr.yaml` to customize OCR behavior:
- Detection model parameters
- Recognition thresholds
- Text classification settings

## Requirements

- Python 3.8+
- OpenCV 4.5+
- RapidOCR
- NumPy
- PyYAML
- Requests (for LLM integration)

## License

This project is open source. Please check individual component licenses.