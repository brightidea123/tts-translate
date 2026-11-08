# Recovered ChatGPT Library snapshots

Historical TTS sources were recovered on November 6, 2026 from authenticated ChatGPT global search for `tts`, using Library download controls. The Library results were reported as May 1 items. These are recovered artifacts, distinct from new reconstruction work.

The byte-for-byte originals are preserved outside the Git checkout in [the Library evidence archive](../../tts-recovery-evidence/library). TTS source SHA-256 checksums, byte sizes, Python AST counts, bytecode headers, and original download paths are recorded in [manifest.json](../../tts-recovery-evidence/library/manifest.json). The archive contains 15 TTS downloads: nine Python source downloads, including one exact duplicate, and six CPython 3.13 bytecode files. A recovered original `apple_vpio_timestamp_bridge.swift` is also archived, bringing the total to 16 downloads / 15 distinct artifacts. Its provenance is in `native-source-provenance.json` and [NATIVE_BRIDGE_RECOVERY.md](NATIVE_BRIDGE_RECOVERY.md). [SHA256SUMS](../../tts-recovery-evidence/library/SHA256SUMS) covers all source and bytecode downloads.

## Comparison baseline

All comparisons below use the original GitHub `main` commit `65be3e4bef2133d5f1be309fe1af42e23c8f6071`, dated May 9, 2026. Its `tts-transcribe.py` is 7,407 lines, with 390 qualified function/class definitions and 23 embedded self-test checks. SHA-256: `02085de5f5a7c0dca56d610616b4b9650965e828e1a8c9ea48f948c73362c1c8`.

The recovered snapshots were parsed with `ast.parse`; their model-loading/application code was not executed. Source inspection and AST comparison show that their core improvements are already present or have later replacements in GitHub HEAD. This establishes the surviving code lineage, but does not establish identical runtime behavior for every historical implementation.

## Recovered source snapshots

| Original download | Lines | Definitions | Checks | Historical behavior and relation to HEAD |
| --- | ---: | ---: | ---: | --- |
| `tts-transcribe.py` | 3,722 | 172 | 0 | Earlier responsive translation/TTS pipeline. Its old `should_preserve_translation_task` and ANSI wrapping helpers have later queue-priority and rendering replacements in HEAD. |
| `tts-transcribe(1).py` | 4,082 | 200 | 0 | Adds timestamped capture, native ring buffer, priority ASR scheduling and ring endpoint detection. HEAD retains these core components, but uses the unified `--apple-bridge-path` instead of this snapshot's separate `--timestamp-bridge-path`. |
| `tts-transcribe_stable_partial_mt.py` | 5,493 | 286 | 5 | Stable partial translation, split partial/final ASR lanes, anti-repeat and hallucination guards, session logs and embedded checks. All named definitions/options/checks survive in HEAD, with later modifications. |
| `tts-transcribe_m4_optimized_review_patch.py` | 5,510 | 287 | 5 | Measures partial-translation quiet time from changes to the stable text, preserves the last sent stable prefix and separates initial-word and delta-word thresholds. HEAD retains these mechanisms. |
| `tts-transcribe_completeness_patch.py` | 5,610 | 289 | 5 | Final/partial transcript completeness merging and sanitizer diagnostics. HEAD reworks the merge into `choose_more_complete_final_text` and `prepare_final_transcript_for_translation`. Its CLI options and checks remain available. |
| `tts-transcribe_language_m4_patch.py` | 5,688 | 302 | 5 | Language specifications and source/target language routing replace hard-coded Turkish/English paths. HEAD retains and extends these. |
| `tts-transcribe_asr_transcribe_guard_patch.py` | 5,777 | 308 | 5 | Strict ASR language and English-only-model guards. All named definitions/options/checks survive in HEAD, with later modifications. |
| `tts-transcribe_guard_helpers_fixed.py` | 5,864 | 311 | 5 | Adds/fixes `is_english_only_whisper_model_name`, `unicode_script_counts`, and `sanitize_asr_language_shape`. All are present in HEAD. This is the largest recovered historical source snapshot. |

`tts-transcribe_completeness_patch (1).py` is an exact duplicate of `tts-transcribe_completeness_patch.py` (both SHA-256 `925d0cfb84d36414ca8593b10629d76433396d9e2ed57c6c1706cd24bfe197ed`) and is retained under its original download name.

The completeness snapshot's directly known Library source is [libfile_02c99112c7bc819185ca8b9a1322b868](https://chatgpt.com/library/file/libfile_02c99112c7bc819185ca8b9a1322b868). Other downloads were obtained through the same authenticated Library search; their exact Library URLs are not yet independently recorded in this manifest. Filenames and download paths are retained without inventing missing identifiers.

## Bytecode-only evidence

Six downloaded `.pyc` files were preserved unchanged. Inspection was restricted to the 16-byte header and readable byte strings; no marshal loading, importing, decompilation, or bytecode execution was used. Their common magic header is recorded in the manifest. Header timestamps are source modification times encoded by the compiler, rather than verified chat creation dates.

| Bytecode download | Header source timestamp (UTC) | Header source bytes | Matching recovered source size |
| --- | --- | ---: | --- |
| `tts-transcribe.cpython-313.pyc` | April 30, 2026, 22:49:25 | 159,306 | No exact source-size match among the recovered `.py` files. |
| `tts-transcribe_stable_partial_mt.cpython-313.pyc` | May 1, 2026, 02:29:47 | 219,670 | `tts-transcribe_stable_partial_mt.py` |
| `tts-transcribe_m4_optimized_review_patch.cpython-313.pyc` | May 1, 2026, 04:08:09 | 220,456 | `tts-transcribe_m4_optimized_review_patch.py` |
| `tts-transcribe_language_m4_patch.cpython-313.pyc` | May 1, 2026, 04:47:17 | 227,345 | `tts-transcribe_language_m4_patch.py` |
| `tts-transcribe_asr_transcribe_guard_patch.cpython-313.pyc` | May 1, 2026, 04:55:50 | 232,193 | `tts-transcribe_asr_transcribe_guard_patch.py` |
| `tts-transcribe_m4_profile_lang_guard_fixed.cpython-313.pyc` | May 1, 2026, 04:59:15 | 233,247 | Corresponding source file has not been recovered. |

Size matches are provenance clues, not proof that bytecode exactly corresponds to a given source. The profile/language-guard bytecode includes readable names for language specifications, ASR guards, model validation, profile application and MLX paths; those feature names are also present in HEAD. Its exact missing source cannot be established through header/string inspection alone.

## Useful differences retained for review

No recovered snapshot provides a confirmed missing improvement that justifies replacing the newer GitHub implementation. The one clearly distinct diagnostic helper is `sanitize_asr_text_with_debug` in the completeness snapshot: it records original and cleaned text when sanitization changes ASR output. HEAD instead reports dropped-ASR reasons and has more extensive language/MLX guards. That historical diagnostic can inform a future focused change if before/after sanitizer traces are needed.

The old timestamp-bridge option refers to a separate capture-helper architecture. HEAD's native helper integration should be restored against HEAD's current raw-PCM/length-prefixed-WAV protocol, rather than reinstating an old option or replacing the main Python source wholesale.
