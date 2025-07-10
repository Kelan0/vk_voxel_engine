use std::fs::File;
use std::io::Read;
use shaderc::{CompilationArtifact, ShaderKind};




fn compile_to_spirv(src: &str, kind: ShaderKind, entry_point_name: &str) -> CompilationArtifact {
    let mut f = File::open(src).unwrap_or_else(|_| panic!("Could not open file {src}"));
    let mut glsl = String::new();
    f.read_to_string(&mut glsl)
        .unwrap_or_else(|_| panic!("Could not read file {src} to string"));

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some(entry_point_name));
    compiler
        .compile_into_spirv(&glsl, kind, src, entry_point_name, Some(&options))
        .expect("Could not compile glsl shader to spriv")
}