use clap::{Arg, Command};
use std::path::Path;
// use compare;

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
}
