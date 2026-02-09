use std::collections::BTreeMap;
use std::fs;
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use image::{DynamicImage, GrayImage, Luma};
use leptess::{LepTess, Variable};

/// OCR Agent — extract text from a scanned PDF using Tesseract OCR.
#[derive(Parser)]
#[command(name = "ocr_agent_rs")]
struct Cli {
    /// Path to the input PDF file
    pdf: PathBuf,

    /// Path to the output text file
    output: PathBuf,

    /// Page spec: e.g. '1-20', '42', '1,5,10-15'. Default: all pages.
    #[arg(long)]
    pages: Option<String>,

    /// Resolution for rendering pages (default: 300). Higher = better but slower.
    #[arg(long, default_value_t = 300)]
    dpi: u32,

    /// Tesseract language code (default: eng).
    #[arg(long, default_value = "eng")]
    lang: String,

    /// Also save individual page text files to this directory.
    #[arg(long)]
    per_page_dir: Option<PathBuf>,
}

/// Parse a page specification like '1,3,5-10,42' into a sorted list of page numbers.
fn parse_page_spec(spec: &str, max_page: usize) -> Result<Vec<usize>> {
    let mut pages = std::collections::BTreeSet::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let mut split = part.splitn(2, '-');
            let start: usize = split
                .next()
                .unwrap()
                .trim()
                .parse()
                .context("invalid page range start")?;
            let end: usize = split
                .next()
                .unwrap()
                .trim()
                .parse()
                .context("invalid page range end")?;
            let start = start.max(1);
            let end = end.min(max_page);
            for p in start..=end {
                pages.insert(p);
            }
        } else {
            let p: usize = part.parse().context("invalid page number")?;
            if p >= 1 && p <= max_page {
                pages.insert(p);
            }
        }
    }
    Ok(pages.into_iter().collect())
}

/// Render a single PDF page to an image::DynamicImage at the given DPI.
fn render_page(doc: &poppler::PopplerDocument, page_idx: usize, dpi: u32) -> Result<DynamicImage> {
    let page = doc
        .get_page(page_idx)
        .context(format!("failed to get page {}", page_idx + 1))?;

    let (pdf_w, pdf_h) = page.get_size();
    let scale = dpi as f64 / 72.0;
    let width = (pdf_w * scale).ceil() as i32;
    let height = (pdf_h * scale).ceil() as i32;

    let mut surface = cairo::ImageSurface::create(cairo::Format::ARgb32, width, height)
        .context("failed to create cairo surface")?;

    let ctx = cairo::Context::new(&surface).context("failed to create cairo context")?;

    // White background
    ctx.set_source_rgb(1.0, 1.0, 1.0);
    ctx.paint().context("failed to paint background")?;

    // Scale and render the page
    ctx.scale(scale, scale);
    page.render(&ctx);

    // Drop the context so we can borrow the surface data
    drop(ctx);

    // Extract pixel data from the cairo surface
    let stride = surface.stride() as usize;
    let data = surface.data().context("failed to get surface data")?;
    let w = width as u32;
    let h = height as u32;

    let mut img = image::RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let offset = y as usize * stride + x as usize * 4;
            // Cairo ARGB32 is stored as native-endian: on little-endian that's BGRA
            let b = data[offset];
            let g = data[offset + 1];
            let r = data[offset + 2];
            let a = data[offset + 3];
            img.put_pixel(x, y, image::Rgba([r, g, b, a]));
        }
    }

    Ok(DynamicImage::ImageRgba8(img))
}

/// Preprocess a scanned page image to improve OCR accuracy.
/// Grayscale → contrast (1.5x) → sharpen → binarize (threshold 140).
fn preprocess_image(img: &DynamicImage) -> GrayImage {
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();

    // Increase contrast by factor 1.5 around mean 128
    let mut contrasted = GrayImage::new(w, h);
    for (x, y, pixel) in gray.enumerate_pixels() {
        let val = pixel[0] as f32;
        let new_val = ((val - 128.0) * 1.5 + 128.0).clamp(0.0, 255.0) as u8;
        contrasted.put_pixel(x, y, Luma([new_val]));
    }

    // Sharpen using a 3x3 kernel:
    //  0 -1  0
    // -1  5 -1
    //  0 -1  0
    let mut sharpened = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                sharpened.put_pixel(x, y, *contrasted.get_pixel(x, y));
                continue;
            }
            let center = contrasted.get_pixel(x, y)[0] as i32;
            let top = contrasted.get_pixel(x, y - 1)[0] as i32;
            let bottom = contrasted.get_pixel(x, y + 1)[0] as i32;
            let left = contrasted.get_pixel(x - 1, y)[0] as i32;
            let right = contrasted.get_pixel(x + 1, y)[0] as i32;
            let val = (5 * center - top - bottom - left - right).clamp(0, 255) as u8;
            sharpened.put_pixel(x, y, Luma([val]));
        }
    }

    // Binarize with threshold 140
    let mut binarized = GrayImage::new(w, h);
    for (x, y, pixel) in sharpened.enumerate_pixels() {
        let val = if pixel[0] > 140 { 255 } else { 0 };
        binarized.put_pixel(x, y, Luma([val]));
    }

    binarized
}

/// Run Tesseract OCR on a preprocessed page image.
fn ocr_image(gray: &GrayImage, lang: &str) -> Result<String> {
    // Encode as PNG in memory for leptess
    let mut png_buf = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(Cursor::new(&mut png_buf));
    gray.write_with_encoder(encoder)
        .context("failed to encode image to PNG")?;

    let mut lt = LepTess::new(None, lang).context("failed to init Tesseract")?;
    lt.set_image_from_mem(&png_buf)
        .context("failed to set image")?;

    lt.set_variable(Variable::TesseditOcrEngineMode, "3")
        .context("failed to set OEM")?;
    lt.set_variable(Variable::TesseditPagesegMode, "6")
        .context("failed to set PSM")?;
    lt.set_variable(Variable::PreserveInterwordSpaces, "1")
        .context("failed to set preserve_interword_spaces")?;

    let text = lt.get_utf8_text().context("failed to get OCR text")?;
    Ok(text)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if !cli.pdf.is_file() {
        bail!("PDF not found: {}", cli.pdf.display());
    }

    println!("PDF: {}", cli.pdf.display());

    let doc = poppler::PopplerDocument::new_from_file(&cli.pdf, None)
        .context("failed to open PDF")?;
    let total_pages = doc.get_n_pages();
    println!("Total pages in PDF: {}", total_pages);

    let pages = if let Some(ref spec) = cli.pages {
        parse_page_spec(spec, total_pages)?
    } else {
        (1..=total_pages).collect()
    };

    if pages.is_empty() {
        bail!("No pages to process");
    }

    println!(
        "Pages to process: {} ({}-{})",
        pages.len(),
        pages.first().unwrap(),
        pages.last().unwrap()
    );
    println!("DPI: {}", cli.dpi);
    println!("Output: {}", cli.output.display());
    if let Some(ref dir) = cli.per_page_dir {
        println!("Per-page dir: {}", dir.display());
    }
    println!();

    if let Some(ref dir) = cli.per_page_dir {
        fs::create_dir_all(dir).context("failed to create per-page directory")?;
    }

    let t_start = Instant::now();
    let total = pages.len();
    let mut results: BTreeMap<usize, String> = BTreeMap::new();

    for (idx, &pg) in pages.iter().enumerate() {
        print!("  [{}/{}] OCR page {}...", idx + 1, total, pg);

        let t0 = Instant::now();

        // Poppler uses 0-based page index
        let img = render_page(&doc, pg - 1, cli.dpi)?;
        let processed = preprocess_image(&img);
        let text = ocr_image(&processed, &cli.lang)?;

        let dt = t0.elapsed().as_secs_f64();
        let line_count = text.trim().lines().count();
        println!(" done ({:.1}s, {} lines)", dt, line_count);

        if let Some(ref dir) = cli.per_page_dir {
            let page_file = dir.join(format!("page_{:04}.txt", pg));
            fs::write(&page_file, &text).context("failed to write per-page file")?;
        }

        results.insert(pg, text);
    }

    let t_total = t_start.elapsed().as_secs_f64();

    // Write combined output
    let separator = "=".repeat(72);
    let mut output = String::new();
    for (&pg, text) in &results {
        output.push('\n');
        output.push_str(&separator);
        output.push('\n');
        output.push_str(&format!("  PAGE {}\n", pg));
        output.push_str(&separator);
        output.push_str("\n\n");
        output.push_str(text);
        output.push('\n');
    }
    fs::write(&cli.output, &output).context("failed to write output file")?;

    let total_chars: usize = results.values().map(|t| t.len()).sum();
    let total_lines: usize = results.values().map(|t| t.trim().lines().count()).sum();

    println!();
    println!(
        "Finished in {:.1}s ({:.1}s per page)",
        t_total,
        t_total / pages.len() as f64
    );
    println!("Extracted {} lines, {} characters", total_lines, total_chars);
    println!("Output written to: {}", cli.output.display());
    if let Some(ref dir) = cli.per_page_dir {
        println!("Per-page files in: {}", dir.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_page_spec_single() {
        let pages = parse_page_spec("5", 100).unwrap();
        assert_eq!(pages, vec![5]);
    }

    #[test]
    fn test_parse_page_spec_range() {
        let pages = parse_page_spec("3-7", 100).unwrap();
        assert_eq!(pages, vec![3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_parse_page_spec_mixed() {
        let pages = parse_page_spec("1,5,10-12,20", 100).unwrap();
        assert_eq!(pages, vec![1, 5, 10, 11, 12, 20]);
    }

    #[test]
    fn test_parse_page_spec_clamped() {
        let pages = parse_page_spec("1-1000", 10).unwrap();
        assert_eq!(pages, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_parse_page_spec_out_of_range_single() {
        let pages = parse_page_spec("50", 10).unwrap();
        assert!(pages.is_empty());
    }

    #[test]
    fn test_parse_page_spec_duplicates() {
        let pages = parse_page_spec("1,1,2,2-4", 10).unwrap();
        assert_eq!(pages, vec![1, 2, 3, 4]);
    }
}
