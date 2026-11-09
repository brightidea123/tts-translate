# Debug-session follow-up: missing translation and short thanks

Reviewed `20261108T212214.763280Z-pid16559-1d909c245e`, launched on November 9,
2026 at 00:22 in Istanbul. This session enabled full debug recording. The
review used saved source audio metadata, request diagnostics, ASR decisions,
transcription/translation logs and shutdown reports. It did not start capture,
restart the user's process, upload audio or run a model.

## Which phrase lacked translation

The logs contain **12 accepted final transcripts and 12 corresponding final
translations**. Utterance 11 was the separate unresolved case; it was not
accepted as final and therefore did not enter committed translation.

Its 8.60-second final request repeatedly produced coherent English recognition
inside a Hebrew → Arabic session. Growing previews from 2.22 to 5.30 seconds
already contained English output. The strict Hebrew native-script check removed
those candidates. Later previews switched to uncertain Hebrew text. The
full-length final again produced English, which was filtered to empty text.

The recently added recovery path ran: its first timestamped window also
recognized English; its second returned two low-confidence Hebrew words. The
candidate did not pass the source-language and consistency checks, so the
decision was `unresolved_final` with `final_recovery_unresolved`. The earlier
provisional words were retained rather than the inconsistent result becoming
an authoritative transcript and translation.

This is a language-selection/filtering failure path, not evidence that a final
translation job vanished. The repeated English candidate is a strong reason to
check for an English-language insert in the source material. Without human
reference speech or independent language verification it cannot establish that
every English hypothesis is correct; Whisper may also generate the wrong
language. A safe correction must verify a language switch acoustically rather
than allowing arbitrary Latin text through the Hebrew guard.

## Where “شكراً جزيلاً” came from

Utterance 14's final source recognition was Hebrew **“תודה רבה.”**, and its
Arabic translation was **“شكراً جزيلاً.”**. The translation preserved the meaning
of the ASR output; any unsupported thank-you message originated upstream in
recognition.

| Saved observation | Utterance 14 |
| --- | ---: |
| Final input duration | 3.00 s |
| Earlier preview durations | 1.42 s, 2.42 s |
| Earlier/final wording | Same two-word Hebrew thank-you |
| Decoder mean log probability | −0.471 |
| Decoder no-speech probability | 0.281 |
| Audio RMS | −19.56 dBFS |
| Clipping / capture-gap flags | None |

Unlike the earlier 20–25-second collapsed finals, this clip had no substantial
conflicting preview. The long-final/preview-consistency guard consequently did
not trigger. Agreement between three decodes is not independent proof that
the words were spoken: overlapping windows can repeat the same model bias.

Utterance 3 similarly recognized a one-word Hebrew thank-you from 1.46 seconds
and translated it as Arabic thanks. Its no-speech probability was higher
(0.634), but its saved waveform also contains substantial energy. Decoder
probabilities and waveform energy cannot by themselves establish whether a
thank-you phrase is genuine, background-media content or a hallucination.

Utterance 14 had no 20 ms frames below −40 dBFS, despite the energy gate
reporting many tiny pauses. Background music or noise can cause that pattern;
it is not a human-labeled measurement of speech. A blanket ban on short thanks
would remove legitimate answers and would not recover missing words.

## Phrase-level language metadata

The web snapshot now accepts a supported canonical `source_language` on an
individual entry, falling back to the session language for absent or invalid
values. Thus an independently verified English insert can be presented and
selected as English → Arabic while surrounding phrases remain Hebrew → Arabic.
The existing web renderer and selection translator already use each entry's
language, so the session's controls and typed-reply direction remain unchanged.
This metadata does not perform or substitute for acoustic language verification.

## Corrections after this review

The MLX final path now requests a fresh decode with **automatic language
detection** when the primary result was removed by the source-script check.
It requires a supported detected language, compatible script, usable decoder
metrics and substantial lexical agreement with the original candidate. Explicit
numbers must agree. A forced Hebrew label is not treated as independent language
evidence, and Latin characters alone cannot select English.

Short generic outro/thank-you finals (up to six seconds) also receive this
additional acoustic check, even without a substantial competing preview. If
the automatic pass gives different wording, another decode forced to the
automatically detected language must corroborate that alternative. The path
adds at most two decodes and reuses the exact prepared audio and original
request identity. It does not acquire new audio or translate the earlier preview
and call that recognition.

Verified foreign-language text is routed to MT using its actual source language;
phrase labels, selection tooltips, history and transcription logs keep that
language. The surrounding session and typed-reply direction stay unchanged.
Native non-outro finals do not incur this extra verification work. Ordinary
short thanks can pass verification; they are not globally blacklisted.

Insufficient quality, disagreement, unsupported language or a decoder failure
leaves the phrase explicitly unverified. The earlier Review needed behavior
preserves provisional text without silently committing a suspect final or
translation. Original and verification decoder outputs remain in diagnostics.

This uses another pass of the same recognition model with different language
selection. It is additional evidence, not an independent human reference or a
guarantee against hallucinations over music/noise. The actual spoken content of
the three-second thank-you clip remains unverified in this audit.

## Shutdown and evidence limits

Graceful Stop completed in 3.556 seconds with no pending final recognition,
unfinished source/translation, queue work or worker callbacks. Missing saved
text should therefore be traced through ASR filtering and unresolved decisions,
not attributed to Stop without supporting evidence.

The session contains **197.62 seconds** of full debug audio. All **140 saved raw
request clips and 140 float32 model inputs** passed their hashes and exact
raw-to-float comparisons. All raw clips matched the captured stream, with no
unmapped requests, capture-gap flags or audit errors. These integrity results
rule out the demonstrated byte-mismatch paths; they do not establish what words
were spoken.

The recording and diagnostics support exact-source replay, but no new model
replay, human transcription, WER/CER score or post-change quality benchmark was
performed in this review. Private conversation wording remains in the ignored
recordings and runtime reports rather than this tracked report.

Validation passed: **352 Python tests, 20 web tests, 35 embedded pipeline checks
and native protocol self-tests**. Added regressions cover automatic versus
forced language provenance, English inserts in Hebrew sessions, normal short
thanks, high-no-speech sign-offs, conflicting numbers, disagreeing confirmation,
bounded failure handling, unchanged audio hashes, per-phrase MT/UI/log direction
and unresolved-final quarantine. No existing process was restarted; a new
session and web-page refresh load these changes.

Local ignored audit: `runtime-reports/he-ar-debug-2026-11-09-integrity.json`.
