use image::imageops::FilterType;
use image::{DynamicImage, ImageError, ImageReader};
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};
use std::path::Path;

const IMG_WIDTH_ONNX_INPUT: usize = 224;
const IMG_HEIGHT_ONNX_INPUT: usize = 224;
const RGB_CHANNEL: usize = 3;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub fn load_image<P: AsRef<Path>>(path: P) -> Result<DynamicImage, ImageError> {
    ImageReader::open(path.as_ref())?.decode()
}

pub fn preprocess(img: &DynamicImage) -> Vec<f32> {
    let img = img
        .resize_exact(
            IMG_WIDTH_ONNX_INPUT as u32,
            IMG_HEIGHT_ONNX_INPUT as u32,
            FilterType::Triangle,
        )
        .to_rgb8();

    let mut tensor = vec![0f32; RGB_CHANNEL * IMG_WIDTH_ONNX_INPUT * IMG_HEIGHT_ONNX_INPUT];

    for y in 0..IMG_HEIGHT_ONNX_INPUT {
        for x in 0..IMG_WIDTH_ONNX_INPUT {
            let p = img.get_pixel(x as u32, y as u32);
            for c in 0..RGB_CHANNEL {
                let mut v = p[c] as f32 / 255.0;
                v = (v - MEAN[c]) / STD[c];

                tensor[c * IMG_WIDTH_ONNX_INPUT * IMG_HEIGHT_ONNX_INPUT
                    + y * IMG_WIDTH_ONNX_INPUT
                    + x] = v;
            }
        }
    }

    tensor
}

pub fn estimate(input: Vec<f32>, model_path: &Path) -> Vec<f32> {
    let input = Array4::from_shape_vec(
        (1, RGB_CHANNEL, IMG_WIDTH_ONNX_INPUT, IMG_HEIGHT_ONNX_INPUT),
        input,
    )
    .unwrap();
    let mut session = Session::builder()
        .unwrap()
        .commit_from_file(model_path)
        .unwrap();

    let outputs = session
        .run(ort::inputs![TensorRef::from_array_view(&input).unwrap()])
        .unwrap();

    let (_shape, feature) = outputs[0].try_extract_tensor::<f32>().unwrap();
    feature.to_vec()
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
