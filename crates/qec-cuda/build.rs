//! Offline SM120 build: `nvcc -arch=sm_120 --cubin --ptxas-options=-v` on the kernel source
//! (0 bytes of VRAM), a check that ptxas reports no register spills, and the C++ driver-API
//! host shim.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().unwrap().parent().unwrap().to_path_buf();
    let cuda_dir = root.join("cuda");
    let kernels = cuda_dir.join("qec_sm120_kernels.cu");
    let params = cuda_dir.join("qec_sm120_params.cuh");
    let shim = manifest.join("native").join("qec_cuda_host.cpp");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cubin = out.join("qec_sm120_kernels.cubin");
    let toolkit = PathBuf::from(std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string()));
    for f in [&kernels, &params, &shim] {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let nvcc = toolkit.join("bin").join("nvcc");
    let output = Command::new(&nvcc)
        .args(["-arch=sm_120", "--cubin", "--ptxas-options=-v"])
        .arg(&kernels)
        .arg("-o")
        .arg(&cubin)
        .output()
        .unwrap_or_else(|e| panic!("cannot run {}: {e}", nvcc.display()));
    let log = format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "nvcc failed:\n{log}");
    // "N bytes stack frame, M bytes spill stores, K bytes spill loads" per function
    let mut entries = 0usize;
    let mut worst_spill = 0u64;
    for line in log.lines() {
        entries += line.contains("Compiling entry function") as usize;
        if let Some(pos) = line.find(" bytes spill stores") {
            let digits: String = line[..pos].chars().rev().take_while(|c| c.is_ascii_digit()).collect();
            let n: u64 = digits.chars().rev().collect::<String>().parse().unwrap_or(0);
            worst_spill = worst_spill.max(n);
        }
    }
    assert!(entries > 0, "ptxas reported no entry functions:\n{log}");
    assert_eq!(worst_spill, 0, "register spill in the SM120 kernels:\n{log}");
    std::fs::write(out.join("ptxas.log"), &log).unwrap();
    println!("cargo:rustc-env=QEC_KERNEL_ENTRIES={entries}");

    cc::Build::new()
        .cpp(true)
        .file(&shim)
        .include(toolkit.join("include"))
        .include(&cuda_dir)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-missing-field-initializers")
        .opt_level(2)
        .compile("qec_cuda_host");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
