//! `vf-translate`: look at the GPU code a file carries, adapt an image offline, or translate supported machine
//! code to sm_70.
//!
//! ```text
//! vf-translate inspect FILE...        fat binaries found in each file (a library, an executable, a raw image)
//! vf-translate extract FILE DIR [SM]  write the decompressed entries of FILE's fat binaries into DIR
//!                             (NNNN.sm_75.ptx, NNNN.sm_86.cubin ...), all or those for architecture SM
//! vf-translate adapt IN OUT           write the image a Volta driver should load in place of IN
//! vf-translate translate IN OUT [MAP [REPORT]]
//!                             write the sm_70 cubin for a supported sm_75 or sm_80 cubin IN; MAP gets one line per
//!                             instruction of the result (function, new offset, old offset, what was
//!                             done), MAP.lives what reference/validate.py reads beside it (where the
//!                             uniform predicates live, what was borrowed), REPORT one line per
//!                             function (`ok` or why it was refused, the name)
//! ```

use std::collections::BTreeMap;
use std::process::ExitCode;

use vf_adapt::{Outcome, Policy};
use vf_fatbin::{self as fatbin, KIND_ELF, KIND_LTO_IR, KIND_PTX};

fn kind_name(kind: u16) -> String {
    match kind {
        KIND_PTX => "PTX".to_string(),
        KIND_ELF => "machine code".to_string(),
        KIND_LTO_IR => "LTO-IR".to_string(),
        other => format!("kind {other}"),
    }
}

/// One line per (kind, architecture) with the number of entries and their stored size.
fn inspect(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let is_text = bytes
        .first()
        .is_some_and(|b| b.is_ascii_graphic() || b.is_ascii_whitespace());
    if let Some(header) = vf_ptx::header(&bytes).filter(|_| is_text) {
        println!(
            "{path}: PTX text, version {}.{}, target sm_{}",
            header.version.0, header.version.1, header.target
        );
        return Ok(());
    }
    let containers = fatbin::find_containers(&bytes);
    if containers.is_empty() {
        match fatbin::cubin_sm(&bytes) {
            Some(sm) if bytes.get(0x12..0x14) == Some(&[0xbe, 0x00]) => {
                println!("{path}: cubin for sm_{sm}")
            }
            _ => println!("{path}: no GPU code found"),
        }
        return Ok(());
    }
    let mut table: BTreeMap<(u16, u32, String), (usize, usize)> = BTreeMap::new();
    for &offset in &containers {
        let fat = fatbin::parse(&bytes[offset..]).map_err(|e| format!("{path}: {e}"))?;
        for entry in &fat.entries {
            let row = table.entry((entry.kind(), entry.sm(), entry.arch_name())).or_default();
            row.0 += 1;
            row.1 += entry.stored.len();
        }
    }
    println!("{path}: {} fat binaries", containers.len());
    for ((kind, _, arch), (count, size)) in table {
        println!(
            "  {:12} {arch:8} {count:6} entries {:10.1} kB",
            kind_name(kind),
            size as f64 / 1e3
        );
    }
    Ok(())
}

/// One file per entry, numbered by fat binary.
fn extract(path: &str, dir: &str, only_sm: Option<u32>) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    let mut written = 0;
    for (index, &offset) in fatbin::find_containers(&bytes).iter().enumerate() {
        let fat = fatbin::parse(&bytes[offset..]).map_err(|e| format!("{path}: {e}"))?;
        let mut occurrences = BTreeMap::new();
        for entry in fat.entries.iter().filter(|e| only_sm.is_none_or(|sm| e.sm() == sm)) {
            let extension = match entry.kind() {
                KIND_PTX => "ptx",
                KIND_ELF => "cubin",
                KIND_LTO_IR => "ltoir",
                _ => "bin",
            };
            let arch = entry.arch_name();
            let occurrence = occurrences.entry((arch.clone(), extension)).or_insert(0usize);
            *occurrence += 1;
            let name = if *occurrence == 1 {
                format!("{dir}/{index:04}.{arch}.{extension}")
            } else {
                format!("{dir}/{index:04}-{occurrence}.{arch}.{extension}")
            };
            let payload = entry
                .payload()
                .map_err(|e| format!("{path}: fat binary {index}: {e}"))?;
            std::fs::write(&name, &payload).map_err(|e| format!("{name}: {e}"))?;
            written += 1;
        }
    }
    println!("{path}: {written} entries written to {dir}");
    Ok(())
}

/// The kernels without code (`vf_adapt::Stubs`): what the layer does with each.
fn push_stubs(stubs: &vf_adapt::Stubs, lines: &mut Vec<String>) {
    for kernel in &stubs.trap_only {
        lines.push(format!(
            "  without code (a trap): {kernel}: given at a lookup, its launch is refused"
        ));
    }
    for kernel in &stubs.empty {
        lines.push(format!("  empty: {kernel}"));
    }
}

fn adapt_bytes(label: &str, bytes: &[u8], policy: &Policy) -> Result<vf_adapt::Adapted, String> {
    if bytes.starts_with(&fatbin::MAGIC.to_le_bytes()) {
        let fat = fatbin::parse(bytes).map_err(|e| format!("{label}: {e}"))?;
        policy.adapt_fatbin(&fat).map_err(|e| format!("{label}: {e}"))
    } else if fatbin::is_elf(bytes) {
        Ok(policy.adapt_cubin(bytes))
    } else {
        Ok(policy.adapt_ptx(bytes))
    }
}

fn format_adapt_lines(label: &str, outcome: &Outcome, device_sm: u32) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    match outcome {
        Outcome::Native => lines.push(format!("{label}: already holds code for sm_{device_sm}; copied")),
        Outcome::Retargeted {
            entries,
            from_sm,
            tensor_rewritten,
            tanh_expanded,
            trapped,
            refused,
            emulated,
            replaced,
            reported,
            stubs,
        } => {
            lines.push(format!(
                "{label}: PTX for sm_{from_sm} retargeted to sm_{device_sm} ({entries} modules, {tensor_rewritten} tensor-core instructions rewritten, {tanh_expanded} tanh expanded, {} functions made to trap, {} instructions in {} functions replaced by reported traps)",
                trapped.len(),
                replaced.iter().map(|function| function.forms.len()).sum::<usize>(),
                replaced.len()
            ));
            for function in trapped {
                lines.push(format!("  trap: {}: {}", function.function, function.reason));
            }
            for kernel in refused {
                lines.push(format!(
                    "  refused kernel: {} ({}: {})",
                    kernel.kernel, kernel.function, kernel.reason
                ));
            }
            for kernel in emulated {
                lines.push(format!(
                    "  emulated in kernel: {}: {}",
                    kernel.kernel,
                    kernel.counts.summary()
                ));
            }
            for function in replaced {
                let mut forms: Vec<&str> = function.forms.iter().map(String::as_str).collect();
                forms.sort_unstable();
                forms.dedup();
                lines.push(format!(
                    "  reported trap: {}: {} instructions ({})",
                    function.function,
                    function.forms.len(),
                    forms.join(", ")
                ));
            }
            for kernel in reported {
                // What the layer gives in a process where no GPU can take part in multicast
                // (`vf_ptx::lower::Options::reported_traps`); elsewhere the kernel is refused.
                lines.push(format!(
                    "  given with reported traps: {}: {}",
                    kernel.kernel,
                    kernel.summary()
                ));
            }
            push_stubs(stubs, &mut lines);
        }
        Outcome::Translated {
            entries,
            from_sm,
            functions,
            refused,
            emulated,
            stubs,
            ..
        } => {
            lines.push(format!(
                "{label}: machine code for sm_{from_sm} translated to sm_{device_sm} ({entries} modules, {functions} functions, {} refused, {} with instructions added)",
                refused.len(),
                emulated.len()
            ));
            for kernel in refused {
                lines.push(format!("  refused: {}: {}", kernel.kernel, kernel.reason));
            }
            push_stubs(stubs, &mut lines);
        }
        Outcome::NotTranslated { from_sm, reason } => {
            return Err(format!(
                "{label}: machine code for sm_{from_sm} that could not be read: {reason}"
            ));
        }
        Outcome::NeedsTranslation { lowest_sm } => {
            return Err(format!(
                "{label}: machine code for sm_{lowest_sm} and later only: needs translation"
            ));
        }
        Outcome::Unknown => return Err(format!("{label}: not a fat binary, a cubin or PTX text")),
    }
    Ok(lines)
}

fn check_retargeted_cubin(bytes: &[u8], policy: &Policy) -> (u8, String) {
    let lines = match adapt_bytes("<worker>", bytes, policy)
        .and_then(|adapted| format_adapt_lines("<worker>", &adapted.outcome, policy.device_sm))
    {
        Ok(lines) => lines,
        Err(message) => return (2, format!("vf-translate: {message}")),
    };
    let Some(header) = lines.first() else {
        return (2, "vf-translate: <worker>: empty outcome".to_string());
    };
    if !header.contains("PTX for sm_80 retargeted") {
        return (3, lines.join("\n"));
    }
    for line in &lines[1..] {
        let s = line.trim();
        if s.starts_with("refused") || s.starts_with("trap:") {
            return (1, s.to_string());
        }
    }
    (0, String::new())
}

fn adapt_worker_io(
    mut reader: impl std::io::Read,
    mut writer: impl std::io::Write,
    policy: &Policy,
) -> Result<(), String> {
    writer
        .write_all(b"VFAW")
        .map_err(|e| format!("adapt-worker stdout: {e}"))?;
    writer.flush().map_err(|e| format!("adapt-worker stdout: {e}"))?;
    let mut len_buf = [0u8; 8];
    loop {
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(format!("adapt-worker stdin: {e}")),
        }
        let len = u64::from_le_bytes(len_buf);
        if len > 512 * 1024 * 1024 {
            return Err(format!("adapt-worker: input too large ({len} bytes)"));
        }
        let mut bytes = vec![0u8; len as usize];
        reader
            .read_exact(&mut bytes)
            .map_err(|e| format!("adapt-worker stdin: {e}"))?;
        let (status, msg) = check_retargeted_cubin(&bytes, policy);
        let msg_bytes = msg.as_bytes();
        let msg_len = u32::try_from(msg_bytes.len()).unwrap_or(u32::MAX);
        writer
            .write_all(&[status])
            .and_then(|()| writer.write_all(&msg_len.to_le_bytes()))
            .and_then(|()| writer.write_all(&msg_bytes[..msg_len as usize]))
            .and_then(|()| writer.flush())
            .map_err(|e| format!("adapt-worker stdout: {e}"))?;
    }
}

fn adapt_worker() -> Result<(), String> {
    let policy = Policy::default().with_environment();
    let stdin = std::io::stdin().lock();
    let stdout = std::io::stdout().lock();
    adapt_worker_io(stdin, stdout, &policy)
}

fn adapt(input: &str, output: &str) -> Result<(), String> {
    let bytes = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
    let policy = Policy::default().with_environment();
    let adapted = adapt_bytes(input, &bytes, &policy)?;
    for line in format_adapt_lines(input, &adapted.outcome, policy.device_sm)? {
        println!("{line}");
    }
    std::fs::write(output, adapted.image.as_deref().unwrap_or(&bytes)).map_err(|e| format!("{output}: {e}"))
}

/// The sm_70 cubin for a supported cubin, with what was refused on standard output.
fn translate(input: &str, output: &str, map: Option<&str>, report: Option<&str>) -> Result<(), String> {
    let bytes = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
    if !vf_fatbin::is_elf(&bytes) {
        return Err(format!("{input}: not a cubin (ELF file)"));
    }
    let opts = vf_sass::translate::Options::from_environment();
    let result = vf_sass::translate::translate_with_options(&bytes, map.is_some(), &opts)
        .map_err(|e| format!("{input}: {e}"))?;
    std::fs::write(output, &result.bytes).map_err(|e| format!("{output}: {e}"))?;
    if let (Some(path), Some(rows)) = (map, &result.map) {
        let mut text = String::new();
        for (function, new, old, what) in rows {
            text.push_str(&format!("{function}\t{new:x}\t{old:x}\t{what}\n"));
        }
        std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))?;
    }
    if let (Some(map), Some(lives)) = (map, &result.lives) {
        // As reference/translate.py writes it (`write_lives`).
        let mut text = String::new();
        for (function, first, second, how, offsets) in lives {
            let offsets: Vec<String> = offsets.iter().map(|at| format!("{at:x}")).collect();
            text.push_str(&format!(
                "{function}\t{first}\t{second}\t{how}\t{}\n",
                offsets.join(",")
            ));
        }
        let path = format!("{map}.lives");
        std::fs::write(&path, text).map_err(|e| format!("{path}: {e}"))?;
    }
    if let Some(path) = report {
        let mut text = String::new();
        for function in &result.report {
            text.push_str(&format!(
                "{}\t{}\n",
                function.refused.as_deref().unwrap_or("ok"),
                function.name
            ));
        }
        std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))?;
    }
    let name = std::path::Path::new(input)
        .file_name()
        .map_or_else(|| input.into(), |n| n.to_string_lossy());
    let mut refused: Vec<(&str, &str)> = result
        .report
        .iter()
        .filter_map(|function| Some((function.name.as_str(), function.refused.as_deref()?)))
        .collect();
    refused.sort();
    println!("{name}: {} functions, {} refused", result.report.len(), refused.len());
    for (function, why) in refused {
        println!("  refused {}: {why}", function.chars().take(60).collect::<String>());
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = match args.as_slice() {
        ["inspect", files @ ..] if !files.is_empty() => files.iter().try_for_each(|file| inspect(file)),
        ["extract", file, dir] => extract(file, dir, None),
        ["extract", file, dir, sm] => match sm.trim_start_matches("sm_").parse() {
            Ok(sm) => extract(file, dir, Some(sm)),
            Err(_) => Err(format!("{sm}: not an architecture (75 or sm_75)")),
        },
        ["adapt", input, output] => adapt(input, output),
        ["adapt-worker"] => adapt_worker(),
        ["translate", input, output] => translate(input, output, None, None),
        ["translate", input, output, map] => translate(input, output, Some(map), None),
        ["translate", input, output, map, report] => translate(input, output, Some(map), Some(report)),
        _ => {
            eprintln!(
                "usage: vf-translate inspect FILE...\n       vf-translate extract FILE DIR [SM]\n       \
                 vf-translate adapt IN OUT\n       vf-translate translate IN OUT [MAP [REPORT]]\n\
                 (the launcher has them as subcommands: vf inspect FILE...)"
            );
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("vf-translate: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vf_fatbin::NewEntry;

    #[test]
    fn extract_preserves_multiple_ptx_entries_for_one_architecture() {
        let fixture = include_bytes!("../../vf-fatbin/tests/data/elf75-ptx75-zstd.fatbin");
        let fat = fatbin::parse(fixture).unwrap();
        let ptx = fat.entries.iter().find(|entry| entry.kind() == KIND_PTX).unwrap();
        let data = fatbin::build(&[
            NewEntry::replace(ptx, b"first".to_vec(), 75, (8, 8)),
            NewEntry::replace(ptx, b"second".to_vec(), 75, (8, 8)),
            NewEntry::replace(ptx, b"third".to_vec(), 80, (8, 8)),
        ]);
        let tmp = std::process::Command::new("mktemp").arg("-d").output().unwrap();
        assert!(tmp.status.success());
        let path = std::path::Path::new(std::str::from_utf8(&tmp.stdout).unwrap().trim());
        let input = path.join("input.fatbin");
        std::fs::write(&input, data).unwrap();
        extract(input.to_str().unwrap(), path.to_str().unwrap(), Some(75)).unwrap();
        assert_eq!(std::fs::read(path.join("0000.sm_75.ptx")).unwrap(), b"first");
        assert_eq!(std::fs::read(path.join("0000-2.sm_75.ptx")).unwrap(), b"second");
        assert!(!path.join("0000.sm_80.ptx").exists());
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn translate_writes_the_lives_beside_the_map() {
        // What reference/translate.py writes for this cubin; reference/validate.py reads it beside
        // the map. (Until 2026-11-22 there was no such file: every uniform predicate "had no life".)
        let cubin = include_bytes!("../../../tests/kernels/translate/k_memory.sm75.cubin");
        let tmp = std::process::Command::new("mktemp").arg("-d").output().unwrap();
        assert!(tmp.status.success());
        let path = std::path::Path::new(std::str::from_utf8(&tmp.stdout).unwrap().trim());
        let input = path.join("k_memory.sm75.cubin");
        std::fs::write(&input, cubin).unwrap();
        let (output, map) = (path.join("out.cubin"), path.join("out.map"));
        translate(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Some(map.to_str().unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(path.join("out.map.lives")).unwrap(),
            "k_generic_index\t0\t2\tfree\tb0,c0,d0,e0,f0,100\nk_generic\t0\t0\tfree\t30,40,50,60,70,80\n"
        );
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn adapt_worker_checks_inputs_without_disk_io() {
        let policy = Policy::default();
        let fixture = include_bytes!("../../vf-fatbin/tests/data/elf75-ptx75-zstd.fatbin");
        let fat = fatbin::parse(fixture).unwrap();
        let ptx = fat.entries.iter().find(|e| e.kind() == KIND_PTX).unwrap();
        let ok_ptx = b".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry k() { ret; }\n";
        let bad_ptx =
            b".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry bad() { barrier.cluster.arrive; ret; }\n";
        let fat_ok = fatbin::build(&[NewEntry::replace(ptx, ok_ptx.to_vec(), 80, (8, 0))]);
        let fat_bad = fatbin::build(&[NewEntry::replace(ptx, bad_ptx.to_vec(), 80, (8, 0))]);
        let fat_75 = fatbin::build(&[NewEntry::replace(
            ptx,
            b".version 8.0\n.target sm_75\n.address_size 64\n.visible .entry k() { ret; }\n".to_vec(),
            75,
            (8, 0),
        )]);

        let empty: Vec<u8> = Vec::new();
        let corrupt = b"\x7fELFgarbage".to_vec();
        let sm75_cubin = include_bytes!("../../../tests/kernels/translate/k_memory.sm75.cubin").to_vec();
        let mut req = Vec::new();
        for item in [&empty, &fat_ok, &corrupt, &fat_bad, &fat_75, &sm75_cubin] {
            req.extend_from_slice(&(item.len() as u64).to_le_bytes());
            req.extend_from_slice(item);
        }

        let mut resp = Vec::new();
        adapt_worker_io(req.as_slice(), &mut resp, &policy).unwrap();
        assert_eq!(&resp[..4], b"VFAW");
        let mut off = 4usize;
        let mut results = Vec::new();
        while off < resp.len() {
            let status = resp[off];
            let mlen = u32::from_le_bytes(resp[off + 1..off + 5].try_into().unwrap()) as usize;
            let msg = std::str::from_utf8(&resp[off + 5..off + 5 + mlen]).unwrap().to_string();
            results.push((status, msg));
            off += 5 + mlen;
        }
        assert_eq!(results.len(), 6);
        assert_eq!(results[0].0, 2);
        assert!(results[0].1.contains("not a fat binary"), "{:?}", results[0]);
        assert_eq!(results[1], (0, String::new()));
        assert_eq!(results[2].0, 2);
        assert!(results[2].1.contains("not a fat binary"), "{:?}", results[2]);
        assert_eq!(results[3].0, 1);
        assert!(results[3].1.starts_with("trap:"), "{:?}", results[3]);
        assert_eq!(results[4].0, 3);
        assert!(results[4].1.contains("PTX for sm_75 retargeted"), "{:?}", results[4]);
        assert_eq!(results[5].0, 3);
        assert!(
            results[5].1.contains("machine code for sm_75 translated"),
            "{:?}",
            results[5]
        );
    }
}
