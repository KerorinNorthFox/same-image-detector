use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageError, ImageReader};

const RGB_CHANNELS: u32 = 3;

pub fn load_image(path: &str) -> Result<DynamicImage, ImageError> {
    ImageReader::open(path)?.decode()
}

pub fn get_image_vec(img: &DynamicImage, width: Option<u32>, height: Option<u32>) -> Vec<f32> {
    let width = width.unwrap_or(256);
    let height = height.unwrap_or(256);
    let resized = img.resize(width, height, FilterType::Triangle);

    let mut vec = Vec::with_capacity((width * height * RGB_CHANNELS) as usize);

    for (_, _, pixel) in resized.pixels() {
        vec.push(pixel[0] as f32 / 255.0);
        vec.push(pixel[1] as f32 / 255.0);
        vec.push(pixel[2] as f32 / 255.0);
    }
    vec
}

pub fn calc_cosine_similarity(v1: &[f32], v2: &[f32]) -> Option<f32> {
    if v1.len() != v2.len() || v1.is_empty() || v2.is_empty() {
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
        return None;
    }

    Some(dot_product / (norm_v1 * norm_v2))
}
