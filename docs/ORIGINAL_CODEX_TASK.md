# Original Codex task recovered

Recovered on November 6, 2026 by opening the old task identifier through Codex's current browser route.

- Exact task title: **Integrate TTS Transcribe version into GitHub**.
- Current URL: https://chatgpt.com/remote/task_e_697f7ad42714832aa862bb22a08e918e
- Earlier URL, retained in both repository pull requests: https://chatgpt.com/codex/tasks/task_e_697f7ad42714832aa862bb22a08e918e
- Associated PR button: **PR 1**, https://github.com/brightidea123/tts-translate/pull/1.
- Scope: one initial user request, execution details, and a completed assistant response. No later revision conversation was present in this task.

## Original user request

The request begins exactly:

> let's start off by integrating the following version into github under tts-transcribe:

It supplies the original Turkish/English Python CLI: mandatory `--mode audio|text`; Turkish speech recognition with Faster Whisper `large-v3` on CPU/int8; NLLB-200 distilled 600M translation with MPS when available; WebRTC VAD; 16 kHz mono Int16 audio in 30 ms frames; English typed replies translated to Turkish and synthesized with Piper's `tr_TR-fahrettin-medium.onnx`; `afplay` playback; terminal Enter/Escape controls. It uses manual key-triggered ASR processing rather than the later concurrent `both`/ring/draft/final implementation.

The complete original user message, including its code, is archived locally at:

[`../../tts-recovery-evidence/task_e_697f7ad42714832aa862bb22a08e918e/original-user-prompt.txt`](../../tts-recovery-evidence/task_e_697f7ad42714832aa862bb22a08e918e/original-user-prompt.txt).

Its UTF-8 SHA-256 is `e6f632290487983522006f7ee93761690433eb51dd564f60ea3144c33d05350e`. The text was checked against the browser's copied user message, including its 7,398 UTF-16 characters and blank-line placement. No hidden model reasoning was extracted.

## Visible execution details

The task searched for `AGENTS.md` in parent directories, listed the repository, and ran `git status -sb`. It then wrote `/workspace/tts-translate/tts-transcribe` using a shell here-document, made it executable with `chmod +x`, checked Git status again, staged the file, and ran:

```sh
git add tts-transcribe
git commit -m 'Add tts-transcribe script'
```

The committed implementation is preserved in original commit `1d7b888`; the obsolete extensionless file was subsequently removed from the current runtime. The parallel duplicate branch contains commit `4fe15a8`. The original Git history and PR evidence independently establish those file versions. The generated file moved the first warning filter after the imports and added blank lines; the archived original prompt preserves the input layout.

The final response reports:

> Added the `tts-transcribe` executable script implementing the audio/text translation modes as provided.

Under Testing it states:

> Not run (not requested).

The UI reports one edited file, `tts-transcribe`, with `+289 -0`. The conversation's follow-up destination is Cloud.

## Relation to later work

This is the February import task linked from the original PRs. It contains no MLX/M4 profile work, multilingual language guards, native ring scheduler, stable partial translation, or later completeness patches. Those developments are evidenced instead by the recovered Library sources and the May 9 GitHub commit. Treating this task link as the missing later development memory would be incorrect.

Accessible active/archived app history, cloud environment listings, current Codex chat search, and all project-linked GitHub comments/releases/workflow records yielded no other later TTS task identifier. This is a limit of the recovered evidence, not proof that later conversations never existed.
