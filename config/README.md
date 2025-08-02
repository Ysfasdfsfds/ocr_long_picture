# Configuration Files

This directory contains configuration files for the OCR Long Picture Processor.

## Files

- `default_rapidocr.yaml`: Default configuration for RapidOCR engine
- `logging.yaml`: Logging configuration (optional)

## Usage

You can specify a custom config file when running the processor:

```bash
python main.py --config config/custom_rapidocr.yaml
```

## Configuration Options

The RapidOCR configuration file supports the following options:

- Detection model settings
- Recognition model settings  
- Classification model settings
- Preprocessing parameters
- Postprocessing parameters

See the RapidOCR documentation for detailed configuration options.