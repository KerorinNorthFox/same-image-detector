use clap::{Arg, Command};
use std::path::Path;
// use compare;

fn main() {
    let matches = Command::new("same-image-detector")
        .arg(Arg::new("base_path").help("").required(true).index(1))
        .arg(Arg::new("target_path").help("").required(true).index(2))
        .get_matches();

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
        None => panic!(),
    };
}
