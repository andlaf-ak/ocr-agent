# OCR Agent

Extracts text from scanned PDF pages using Tesseract OCR. Includes image preprocessing (grayscale conversion, contrast enhancement, sharpening, binarization) to improve accuracy on scanned documents.

## Prerequisites

- Rust toolchain (1.70+)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and [Leptonica](http://www.leptonica.org/) installed
- [Poppler](https://poppler.freedesktop.org/) with GLib bindings
- [Cairo](https://cairographics.org/)

### Install system dependencies (macOS)

```bash
brew install tesseract poppler cairo
```

## Build

```bash
cd ocr_agent_rs
cargo build --release
```

## Usage

```bash
./target/release/ocr_agent_rs <input.pdf> <output.txt> [options]
```

### Examples

```bash
# Extract all pages
./target/release/ocr_agent_rs document.pdf document.txt

# Extract a specific page range
./target/release/ocr_agent_rs document.pdf document.txt --pages 1-20

# Extract a single page
./target/release/ocr_agent_rs document.pdf document.txt --pages 42

# Extract specific pages (comma-separated)
./target/release/ocr_agent_rs document.pdf document.txt --pages 1,5,10-15,42

# Higher DPI for better quality (slower)
./target/release/ocr_agent_rs document.pdf document.txt --dpi 400

# Save individual per-page text files
./target/release/ocr_agent_rs document.pdf document.txt --per-page-dir pages/

# Use a different Tesseract language
./target/release/ocr_agent_rs document.pdf document.txt --lang ita
```

### Options

| Option | Default | Description |
|---|---|---|
| `--pages` | all | Page spec: `1-20`, `42`, `1,5,10-15` |
| `--dpi` | 300 | Render resolution. Higher = better but slower |
| `--lang` | `eng` | Tesseract language code |
| `--per-page-dir` | none | Directory to save individual page text files |
