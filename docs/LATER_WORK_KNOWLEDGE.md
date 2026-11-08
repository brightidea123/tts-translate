# October and later TTS work: knowledge recovery

Updated November 6, 2026. This document separates the user's confirmed recollections from independently recovered records. The restored May CLI is an older baseline, not a recovery of the October web application.

## Confirmed by the user in the recovery conversation

The following facts come directly from the user on November 6, 2026; they have not been recovered from the missing October conversations:

- Substantially more work took place in **October 2026**, possibly continuing later.
- Work was mainly performed **directly in the project directory**, without necessarily copying/pasting or uploading code. Uploaded Library snapshots therefore cannot establish the final state.
- The later application had a **web UI**. The terminal dashboard in the May source does not satisfy that interface requirement.
- Later work involved **LitLink as an audio source**.
- Additional languages included **Hebrew**.
- The GUI handled language direction correctly. The user's original wording was "a GUI for handling left-to-right languages correctly"; retain that wording rather than silently replacing it with an assumed RTL-only requirement. Exact requirements for mixed-direction paragraphs, alignment, punctuation, numbers, selection/copying, and per-field direction have not been recovered.
- The recovery priority is **chat communication, requirements, decisions, and refinements**, even if the exact later code cannot be recovered.

These are requirements/context to retain for continuation. They are not evidence that the current restored code implements the October version, nor a substitute for the lost conversations.

## Independently recovered, but older

- GitHub's saved `main` is dated May 9, 2026 and contains a terminal-based Python application, including Hebrew language metadata and Unicode/script handling.
- The recovered Library sources are historical May variants. No recovered source snapshot contains the later web UI or a LitLink integration.
- The recovered Codex task **Integrate TTS Transcribe version into GitHub** is the original February import, not an October development conversation. See [ORIGINAL_CODEX_TASK.md](ORIGINAL_CODEX_TASK.md).
- The raw native Swift bridge added during restoration is a new reconstruction from an older recovered helper and the saved Python protocol. It is not recovered October work.

## Current LitLink integration clues

Read-only inspection found `/Applications/LitLink.app`, bundle `com.audiobridge.app`, version 2.9.14 (108), plus `/Library/Audio/Plug-Ins/HAL/AudioBridgeDriver.driver`. The app's routing identifiers include `LitLink Audio Bridge`, tracks 1–8, `AudioBridge_UID`, and `AudioBridge_UID_Track_N`. Its bundle registers `litlink://`.

These identify the currently installed audio-routing product and possible input-device names. They do not establish which track, channel layout, capture method, or routing configuration the October TTS application used. No old TTS project path, URL, or requirements record was found in the inspected LitLink metadata/resource/log locations. Current preferences were modified on November 6 and cannot establish the pre-reset setup.

## Search coverage and limits

- Local Codex session/history indexes contain the November 6 recovery work, without pre-reset sessions; the local generated-memory database has no prior memory entries.
- The app's accessible ChatGPT listing exhausted at 140 chats, July 2–November 5, including 68 updated October 1 or later. Five plausible recent technical conversations (17 available turns) contained no related TTS, web-UI, Hebrew, or LitLink refinements. This was a targeted candidate review, not a claim that every unrelated conversation body was read.
- Accessible active/archived app history and returned hosts supplied no matching later Codex conversation. Browser cloud archives and search were also inspected; the known February task remains reachable by its direct current URL.
- Browser global search for `LitLink` returned no results; `Hebrew` under Chats also returned none. Library search for `LitLink` found no file; `gui` results were unrelated guide/game-engine assets.
- The visible memory summary and legacy saved memories contained no TTS project requirement. No memory settings were changed.
- The saved **Translation: English <> Turkish** project has no chats and contains instructions for strict English/Turkish text translation, rather than TTS development context.
- GitHub inspection found no later public successor under `brightidea123`, and no matching LitLink/AudioBridge project in the connected account's repositories. Search indexing for the old repository is incomplete, so negative code-search results do not exclude a private or unindexed successor.

No October-or-later TTS conversation, web-UI source, or detailed refinement history has been recovered. These access limits do not establish that the conversations never existed. Further recovery needs an additional surviving history source, account/workspace, backup, export, or concrete task identifier; the details must not be invented from the May implementation.

## Still unknown

Web framework and frontend/backend structure; application name and later path; exact UI layout and controls; Hebrew and mixed-script rendering rules; LitLink track/device selection; browser-to-backend audio transport; model/backend choices made in October; streaming/final revision behavior in the web UI; performance targets; and the later bug reports, accepted fixes, and rejected approaches.

Do not build a guessed replacement and label those choices recovered decisions. Keep future independently found requirements traceable to their conversation or artifact.
