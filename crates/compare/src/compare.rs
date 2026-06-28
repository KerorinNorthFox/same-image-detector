use image::imageops::FilterType;
use image::{DynamicImage, ImageError, ImageReader};

const NORMALIZED: f32 = 1.0 / 255.0;

pub fn load_image(path: &std::path::PathBuf) -> Result<DynamicImage, ImageError> {
    ImageReader::open(path)?.decode()
}

pub fn get_image_vec(img: &DynamicImage, width: Option<u32>, height: Option<u32>) -> Vec<f32> {
    let width = width.unwrap_or(256);
    let height = height.unwrap_or(256);
    let resized = img.resize_exact(width, height, FilterType::Nearest);
    let rgb = resized.to_rgb8();
    let raw = rgb.as_raw();

    let mut vec = vec![0.0; raw.len()];

    for (dst, src) in vec.iter_mut().zip(raw) {
        *dst = *src as f32 * NORMALIZED;
    }
    vec
}

pub fn calc_cosine_similarity(v1: &[f32], v2: &[f32]) -> Option<f32> {
    if v1.len() != v2.len() || v1.is_empty() || v2.is_empty() {
        dbg!("v1 or v2 is empty");
        return None;
    }

    let mut dot_product = 0.0;
    let mut norm_v1 = 0.0;
    let mut norm_v2 = 0.0;

    for (a, b) in v1.iter().zip(v2.iter()) {
        dot_product += a * b;
        norm_v1 += a * a;
        norm_v2 += b * b;
    }

    if norm_v1 == 0.0 || norm_v2 == 0.0 {
        dbg!(norm_v1);
        dbg!(norm_v2);
        return None;
    }

    Some(dot_product / (norm_v1.sqrt() * norm_v2.sqrt()))
}
