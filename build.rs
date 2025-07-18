use std::fs;
use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;
const SDL_LIB_DIR: &str = "E:\\Code\\Libraries\\SDL3-3.2.16\\VC\\lib\\x64";
const SDL_DLL_PATH: &str = "E:\\Code\\Libraries\\SDL3-3.2.16\\VC\\lib\\x64\\SDL3.dll";

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.ancestors().nth(3).unwrap();

    copy_file(SDL_DLL_PATH, target_dir.to_str().unwrap(), "dll");

    println!("cargo::rustc-link-search={SDL_LIB_DIR}");
    // println!("cargo:rustc-env=RUSTFLAGS=--cfg ");
}

fn copy_file(source_file: &str, target_dir: &str, extension_type: &str) -> bool {
    let src = PathBuf::from(source_file);
    let mut dst = PathBuf::from(target_dir);
    if !src.is_file() || !dst.is_dir() {
        return false;
    }

    let _extension = match src.extension() {
        Some(val) if val == OsStr::new(extension_type) => val,
        _ => return false,
    };

    let file_name = match src.file_name() {
        Some(val) => val,
        None => return false
    };

    dst.push(file_name);

    fs::copy(src, &dst).is_ok()
}