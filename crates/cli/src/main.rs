use clap::{Arg, Command};
use compare;
use std::fs;
use std::path::{Path, PathBuf};

// ディレクトリから画像ファイルを抽出
fn get_img_paths(dir: fs::ReadDir) -> Vec<PathBuf> {
    dir.filter_map(|entry| entry.ok()) // エラーのないファイルだけ
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && does_contains_img_ext(path)) // 画像ファイルだけを抽出
        .collect()
}

// 画像の拡張子がファイルパスに含まれているか
fn does_contains_img_ext(path: &PathBuf) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp"
            )
        })
        .unwrap_or(false)
}

fn main() {
    let matches = Command::new("same-image-detector")
        .arg(Arg::new("base_path").help("Comparison source directory. (and target directory if target_path does not exist.)").required(true).index(1))
        .arg(Arg::new("target_path").help("Comparison target directory.").required(false).index(2))
        .get_matches();

    // 比較元ディレクトリ.
    // 比較崎がない場合、比較元同士を比較する.
    let base_path = match matches.get_one::<String>("base_path") {
        Some(base_path_arg) => {
            dbg!(base_path_arg);
            Path::new(base_path_arg)
        }
        None => panic!(),
    };
    let target_path = match matches.get_one::<String>("target_path") {
        Some(target_path_arg) => {
            dbg!(target_path_arg);
            Path::new(target_path_arg)
        }
        None => {
            println!("Detect in same directory.");
            base_path
        }
    };

    if base_path.is_file() {
        panic!(
            "base_path {} is file. Use direcotry path.",
            base_path.display()
        );
    }
    if target_path.is_file() {
        panic!(
            "target_path {} is file. Use direcotry path.",
            target_path.display()
        );
    }

    let base_dir = fs::read_dir(base_path).unwrap();
    let target_dir = fs::read_dir(target_path).unwrap();
    let base_img_paths = get_img_paths(base_dir);
    let target_img_paths = get_img_paths(target_dir);

    let base_vecs: Vec<_> = base_img_paths
        .iter()
        .map(|path| {
            dbg!(path);
            let img = compare::load_image(path).unwrap();
            compare::get_image_vec(&img, None, None)
        })
        .collect();
    let target_vecs: Vec<_> = target_img_paths
        .iter()
        .map(|path| {
            dbg!(path);
            let img = compare::load_image(path).unwrap();
            compare::get_image_vec(&img, None, None)
        })
        .collect();
    dbg!("Image conversion is completed.");

    for base_vec in &base_vecs {
        for target_vec in &target_vecs {
            let result = compare::calc_cosine_similarity(&base_vec, &target_vec);
            dbg!(result);
        }
    }
}
