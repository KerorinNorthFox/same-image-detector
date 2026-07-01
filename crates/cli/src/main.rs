use clap::{Arg, Command};
use compare;
use rayon::prelude::*;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const THRESHOULD: i32 = 8;
const IMG_WIDTH: u32 = 10;
const IMG_HEIGHT: u32 = 10;

struct ImageFeature {
    path: PathBuf,                 // 画像のパス.
    hash: usize,                   // 画像のハッシュ.
    is_move: bool,                 // 移動先が決まっているか.
    move_to_path: Option<PathBuf>, // 移動先のパス.
}

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

// ファイルを移動する.
// 移動先のディレクトリが存在しない場合、ネストを含め全てのディレクトリを作成する.
// from: 移動元ファイル(ファイル名を含める).
// to  : 移動先ファイル(ファイル名を含める).
fn move_file(from: &Path, to: &Path) -> io::Result<()> {
    let parent = to.parent().unwrap();
    if !parent.exists() {
        fs::create_dir_all(parent)?;
        println!("create dir '{}' successfully.", parent.to_str().unwrap());
    }

    fs::rename(&from, &to)?;
    Ok(())
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

    // 画像を全てベクトルに変換する.
    let mut base_features: Vec<_> = base_img_paths
        .par_iter()
        .map(|path| {
            let img =
                compare::load_image(path).expect(&format!("'{}' failed to load.", path.display()));
            let hash = compare::get_hash(&img, Some(IMG_WIDTH), Some(IMG_HEIGHT));
            ImageFeature {
                path: dbg!(path.clone()),
                hash: hash,
                is_move: false,
                move_to_path: None,
            }
        })
        .collect();
    let mut target_features: Vec<_> = target_img_paths
        .par_iter()
        .map(|path| {
            let img =
                compare::load_image(path).expect(&format!("'{}' failed to load.", path.display()));
            let hash = compare::get_hash(&img, Some(IMG_WIDTH), Some(IMG_HEIGHT));
            ImageFeature {
                path: dbg!(path.clone()),
                hash: hash,
                is_move: false,
                move_to_path: None,
            }
        })
        .collect();
    dbg!("Image conversion is completed.");

    // 重複画像の移動先は '比較元ディレクトリ/#duplicated' にする.
    let save_dir_path = base_path.join("#duplicated");

    // 重複画像のImageFeatureに移動先のパスを追加する.
    for base_feat in &mut base_features {
        let base_img_path = &base_feat.path;
        // 重複画像を分けるために比較元画像ファイル名でディレクトリを分ける.
        let save_unique_dir = save_dir_path.join(base_img_path.file_stem().unwrap());
        dbg!(&base_img_path);

        for target_feat in &mut target_features {
            let target_img_path = &target_feat.path;
            dbg!(&target_img_path);

            let sim = compare::calc_distance(base_feat.hash, target_feat.hash);
            dbg!(sim);

            // 類似度が閾値を下回る場合.
            if sim < THRESHOULD {
                println!("These images is similar.");
                target_feat.is_move = true;
                target_feat.move_to_path =
                    Some(save_unique_dir.join(target_img_path.file_name().unwrap()));

                if !base_feat.is_move {
                    base_feat.is_move = true;
                    base_feat.move_to_path =
                        Some(save_unique_dir.join(base_img_path.file_name().unwrap()));
                }
            }
        }
    }

    base_features.par_iter().for_each(|feat| {
        if !feat.is_move {
            return;
        }

        if let Some(move_to_path) = &feat.move_to_path {
            match move_file(dbg!(&feat.path), dbg!(&move_to_path)) {
                Err(e) => {
                    dbg!(e);
                }
                Ok(_) => {
                    println!(
                        "Move {} is completed successfully.",
                        feat.path.to_str().unwrap()
                    );
                }
            }
        }
    });
    target_features.par_iter().for_each(|feat| {
        if !feat.is_move {
            return;
        }

        if let Some(move_to_path) = &feat.move_to_path {
            match move_file(dbg!(&feat.path), dbg!(&move_to_path)) {
                Err(e) => {
                    dbg!(e);
                }
                Ok(_) => {
                    println!(
                        "Move {} is completed successfully.",
                        feat.path.to_str().unwrap()
                    );
                }
            }
        }
    });
}
