use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageError, ImageReader};
use std::path::Path;

pub fn load_image<P: AsRef<Path>>(path: P) -> Result<DynamicImage, ImageError> {
    ImageReader::open(path.as_ref())?.decode()
}

pub fn get_hash(img: &DynamicImage, width: Option<u32>, height: Option<u32>) -> usize {
    let width = width.unwrap_or(256);
    let height = height.unwrap_or(256);
    let grayscale = img
        .resize_exact(width, height, FilterType::Lanczos3)
        .grayscale();

    let mut sum_pixels: usize = 0;
    let mut pixels: Vec<usize> = Vec::new();

    for (_x, _y, pixel) in grayscale.pixels() {
        let red = pixel[0];
        sum_pixels += red as usize;
        pixels.push(red as usize);
    }

    let (width, height) = grayscale.dimensions();

    // 画素値の平均値を取得
    let ave = (sum_pixels as f64) / (f64::from(width) * f64::from(height));

    let mut hash: usize = 0;
    let mut one: usize = 1;

    // Average hash の計算
    for pixel in pixels {
        if pixel as f64 > ave {
            hash |= one;
        }
        one <<= 1
    }
    hash
}

pub fn calc_distance(hash_from: usize, hash_to: usize) -> i32 {
    let mut d = 0;
    for i in 0..64 {
        let k = 1 << i;
        if (hash_from & k) != (hash_to & k) {
            d += 1
        }
    }
    d
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
