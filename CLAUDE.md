# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese OCR (Optical Character Recognition) project focused on processing long images, particularly chat screenshots from messaging apps. The system implements intelligent image slicing, OCR recognition, chat message analysis, and AI-powered querying capabilities.

## Key Commands

### Setup and Installation
```bash
# Install Python dependencies
pip install -r python/requirements.txt

# For different OCR engines, use specific requirement files:
pip install -r python/requirements_ort.txt        # ONNXRuntime backend
pip install -r python/requirements_paddle.txt    # PaddlePaddle backend
pip install -r python/requirements_torch.txt     # PyTorch backend
pip install -r python/requirements_vino.txt      # OpenVINO backend
```

### Running OCR Processing
```bash
# Process a long image with chat analysis
python long_image_ocr_opencv.py

# Run basic RapidOCR demo
cd python && python demo.py

# Run RapidOCR with command line options
cd python && python -m rapidocr --img_path IMAGE_PATH --text_score 0.5 --vis_res

# Run tests
cd python && python -m pytest tests/
```

### Web Interface Options
```bash
# Single-user web interface
cd ocrweb && python rapidocr_web/ocrweb.py

# Multi-user web interface
cd ocrweb_multi && python main.py
```

### Docker Operations
```bash
# Build and run with Docker
cd docker && ./docker_build&run.sh

# Stop and clean Docker containers
cd docker && ./docker_stop&clean.sh
```

### LLM Integration
```bash
# Start interactive chat analysis (requires Ollama)
python LLM_run.py
```

## Architecture Overview

### Core Components

1. **Main Processing Pipeline** (`long_image_ocr_opencv.py`)
   - Long image slicing with configurable overlap
   - Multi-engine OCR processing (ONNXRuntime, PaddlePaddle, PyTorch, OpenVINO)
   - Chat message structure analysis and reconstruction
   - Avatar detection and positioning
   - Content deduplication across slices

2. **RapidOCR Engine** (`python/rapidocr/`)
   - Modular OCR system with pluggable backends
   - Text detection (`ch_ppocr_det/`)
   - Text classification (`ch_ppocr_cls/`)
   - Text recognition (`ch_ppocr_rec/`)
   - Configurable via YAML files

3. **Avatar Processing** (`process_avatar.py`)
   - Computer vision-based avatar detection
   - Contour analysis and geometric filtering
   - Coordinate system transformation between slices and original image

4. **LLM Integration** (`LLM_run.py`)
   - Ollama API integration for chat analysis
   - Structured prompt engineering for chat context understanding
   - Interactive Q&A capabilities

### Data Flow

1. **Image Input** → **Slice Generation** → **OCR Processing** → **Avatar Detection**
2. **Content Analysis** → **Deduplication** → **Chat Reconstruction** → **LLM Integration**
3. **Output**: Structured JSON with chat messages, timestamps, and speaker identification

### Configuration System

- **Main Config**: `default_rapidocr.yaml` - Global OCR settings
- **Engine Configs**: `python/rapidocr/config.yaml` - RapidOCR engine parameters
- **Model Configs**: Various YAML files for different OCR models and languages

### Output Structure

The system generates:
- `output_images/`: Processed image slices and visualization results
- `output_json/`: Structured data including OCR results and chat analysis
- Interactive JSON format compatible with LLM processing

## Development Guidelines

### Key Classes and Methods

- `LongImageOCR`: Main processing class with slice management
- `RapidOCR`: Core OCR engine with multi-backend support
- `process_avatar()`: Avatar detection and coordinate transformation
- `process_with_llm()`: LLM integration for chat analysis

### Testing

Run tests from the python directory:
```bash
cd python
python tests/test_main.py          # Core functionality
python tests/test_ort.py           # ONNXRuntime backend
python tests/test_paddle.py        # PaddlePaddle backend
```

### Model Management

Models are automatically downloaded and cached in:
- `python/rapidocr/models/`
- `python/rapidocr_*/models/` for different backends

### Performance Considerations

- Image slicing reduces memory usage for large images
- Configurable overlap prevents text loss at slice boundaries
- Multi-threading support in inference engines
- Caching mechanisms for repeated model usage

## Platform Support

- **Languages**: Chinese (primary), English, Japanese, Korean
- **Platforms**: Windows, Linux, macOS
- **Deployment**: Docker, web interfaces, Python package
- **Mobile**: Android and iOS implementations available

## Integration Points

- **Web Services**: REST API through web interfaces
- **Command Line**: Full CLI support with argument parsing
- **Python Package**: Installable via setup.py
- **Docker**: Containerized deployment options
- **LLM Backends**: Ollama integration (extensible to other LLM providers)