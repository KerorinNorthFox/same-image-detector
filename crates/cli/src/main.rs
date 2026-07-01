use clap::{Arg, Command};
use compare;
use ort::ep;
use ort::ep::ExecutionProvider;
use ort::session::Session;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const THRESHOULD: f32 = 0.8;
const MODEL_NAME: &str = "model/resnet50_feature.onnx";

struct ImageFeature {
    vec: Vec<f32>,                 // 画像のベクトル.
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
        println!("Created dir '{}' successfully.", parent.to_str().unwrap());
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

    // バイナリからの相対的なモデルパス.
    let exe_path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let mut model_path = exe_path.join(MODEL_NAME);
    if cfg!(debug_assertions) {
        model_path = Path::new(MODEL_NAME).to_path_buf();
    }
    dbg!(&model_path);

    // スレッド数-2でrayonのスレッドプールをセットアップする.
    let threads = dbg!(std::thread::available_parallelism())
        .unwrap()
        .get()
        .saturating_sub(4);
    ThreadPoolBuilder::new()
        .num_threads(dbg!(threads.max(1))) // 最低でも1つのスレッドを使用.
        .build_global()
        .unwrap();
    let cuda = ep::CUDA::default();
    dbg!(cuda.is_available().unwrap());

    // 画像を全てベクトルに変換する.
    let mut base_features: HashMap<PathBuf, ImageFeature> = base_img_paths
        .par_iter()
        .map_init(
            // 各スレッドごとにortのsessionを用意し使いまわす.
            || {
                Session::builder()
                    .unwrap()
                    .with_execution_providers([ep::CUDA::default().build()])
                    .unwrap()
                    .with_intra_threads(1) // session内部のスレッドは1つだけ.
                    .unwrap()
                    .commit_from_file(&model_path)
                    .expect(&format!("Failed to load model '{}'.", model_path.display()))
            },
            |session, path| {
                let img = compare::load_image(path)
                    .expect(&format!("Failed to load image '{}'.", path.display()));
                let vec = compare::preprocess(&img);
                let est_vec = compare::estimate(session, vec);
                let feature = ImageFeature {
                    vec: est_vec,
                    is_move: false,
                    move_to_path: None,
                };
                (dbg!(path.clone()), feature)
            },
        )
        .collect();
    let mut target_features: HashMap<PathBuf, ImageFeature> = target_img_paths
        .par_iter()
        .map_init(
            || {
                Session::builder()
                    .unwrap()
                    .with_intra_threads(1)
                    .unwrap()
                    .commit_from_file(&model_path)
                    .expect(&format!("Failed to load model '{}'.", model_path.display()))
            },
            |session, path| {
                let img = compare::load_image(path)
                    .expect(&format!("Failed to load image '{}'.", path.display()));
                let vec = compare::preprocess(&img);
                let est_vec = compare::estimate(session, vec);
                let feature = ImageFeature {
                    vec: est_vec,
                    is_move: false,
                    move_to_path: None,
                };
                (dbg!(path.clone()), feature)
            },
        )
        .collect();
    println!("Image conversion is completed.");

    // 重複画像の移動先は '比較元ディレクトリ/#duplicated' にする.
    let save_dir_path = base_path.join("#duplicated");

    // 重複画像のImageFeatureに移動先のパスを追加する.
    for (base_img_path, base_feat) in &mut base_features {
        // 重複画像を分けるために比較元画像ファイル名でディレクトリを分ける.
        let save_unique_dir = save_dir_path.join(base_img_path.file_stem().unwrap());

        // targetの方で既にis_moveフラグが立っているbaseはスキップ.
        if target_features.get(base_img_path).unwrap().is_move {
            println!(
                "'{}' is already flaged is_move on target_feat.",
                base_img_path.display()
            );
            continue;
        }

        for (target_img_path, target_feat) in &mut target_features {
            // 同じファイルの時はスキップ.
            if base_img_path == target_img_path {
                println!(
                    "{} == {}.",
                    base_img_path.display(),
                    target_img_path.display()
                );
                continue;
            }

            // targetのis_moveフラグが既に立っている場合スキップ.
            if target_feat.is_move {
                println!("'{}' is already flaged is_move.", target_img_path.display());
                continue;
            }

            let result = compare::calc_cosine_similarity(&base_feat.vec, &target_feat.vec);
            println!(
                "{} <-> {} = {:?}",
                base_img_path.display(),
                target_img_path.display(),
                result
            );

            // 類似度が閾値を上回る場合.
            if let Some(sim) = result
                && sim > THRESHOULD
            {
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

    base_features.iter().for_each(|(path, feat)| {
        if !dbg!(feat.is_move) {
            return;
        }

        if let Some(move_to_path) = &feat.move_to_path {
            match move_file(dbg!(&path), dbg!(&move_to_path)) {
                Err(e) => {
                    dbg!(e);
                }
                Ok(_) => {
                    println!("'{}' is moved successfully.", path.to_str().unwrap());
                }
            }
        }
    });
    target_features.iter().for_each(|(path, feat)| {
        if !dbg!(feat.is_move) {
            return;
        }

        if let Some(move_to_path) = &feat.move_to_path {
            match move_file(dbg!(&path), dbg!(&move_to_path)) {
                Err(e) => {
                    dbg!(e);
                }
                Ok(_) => {
                    println!("'{}' is moved successfully.", path.to_str().unwrap());
                }
            }
        }
    });
}
