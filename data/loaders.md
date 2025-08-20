# Data Loading Requirements

## Supported Formats
- HEIC (iPhone default) - use pillow-heif
- JPEG, PNG
- 6 trained classes (5 diseases + Healthy)
- Unknown is NOT a class - it's uncertainty-based

## Requirements
- Handle iPhone HEIC format
- Support batch loading
- Apply preprocessing pipeline

Related: /preprocessing/pipeline.md