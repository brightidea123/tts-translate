# Native audio source recovery and reconstruction

The original **timestamped capture helper** was recovered from the account Library as `apple_vpio_timestamp_bridge.swift`, downloaded to `~/Downloads`, and copied byte-for-byte into this repository. Its SHA-256 is:

```text
627f8264f0e425ca9752e148dbbcf3d32a432504f2b701cdb7cff88b29b482a2
```

The downloaded file and the repository copy have the same hash. This is a recovered source artifact, not evidence that it was part of the saved Git commit or the final working local installation. It emits `APCM` timestamped capture frames and has no stdin playback implementation.

The active Python `AppleVPIOBridge` expects a different, raw-duplex protocol. The missing original `apple_vpio_bridge.swift` was **not recovered**. The file with that name now supplied here is an explicitly labeled **reconstructed derivative**, adapted from the recovered helper's AVAudioEngine capture/conversion approach and the Python client's actual protocol. It must not be represented as the original missing file.

## Reconstructed implementation

The reconstructed helper supports:

- `APPLE_VPIO_MODE=raw|vpio`; VPIO setup failures attempt a raw fallback.
- `APPLE_VPIO_AGC=on|off` and `APPLE_VPIO_BYPASS=on|off` for voice processing, with verbose/debug environment switches.
- Mono 16 kHz signed Int16 little-endian capture on stdout, buffered into 320-sample frames, without any `APCM` headers.
- Repeated stdin messages containing a four-byte little-endian WAV byte count followed by a complete RIFF/WAVE file. Reads can split headers or payloads across arbitrary boundaries.
- WAV validation/decoding using AVAudioFile, conversion to the engine playback format, and sequential scheduling through AVAudioPlayerNode.
- Stderr readiness beginning `apple-vpio-bridge ready`, emitted after engine startup; the exact no-input diagnostic expected by Python after two seconds without converted capture frames.
- Signal/stdin-EOF shutdown and explicit errors for malformed/truncated input. A single WAV message is limited to 64 MiB.
- Bounded capture-output work (four pending chunks; new chunks are dropped if stdout stalls) and at most two scheduled playback buffers, with stdin backpressure while playback catches up.

The complete Python-side contract, including the separate timestamped format, is in [`RECOVERED_DESIGN.md`](RECOVERED_DESIGN.md). The Python core was not changed to consume the timestamped protocol.

## Build and offline verification

With Apple's command-line tools available:

```sh
make build-native
make check-native
```

`make build-native` compiles the reconstructed raw helper with AVFoundation. The resulting `apple_vpio_bridge` binary is ignored by Git. `make check-native` invokes `--self-test`, which branches before creating AVAudioEngine and does not request microphone access or play audio.

The reconstructed helper compiled and all offline checks passed during restoration. The test uses the production raw encoder/framer with signed edge samples spanning callback/frame boundaries. It parses two consecutive real WAV messages one byte at a time, decodes a known PCM fixture through the actual AVAudioFile path, checks decoded sample values, and exercises the actual playback format converter, including mono 16 kHz to stereo 48 kHz conversion. It also checks rejection of oversized, truncated, and non-WAV input. These checks validate the reconstructed protocol implementation; they do not validate physical audio capture or playback. The current SDK emits deprecation warnings for legacy AVAudioEngine connect/tap/play APIs.

The recovered timestamped helper remains unchanged and passed Swift type checking. It retains warnings about an unnecessary `try`/unreachable `catch` around conversion, plus a tap API deprecation. It can be compiled separately if needed, but its stdout is incompatible with the active raw bridge reader; renaming its binary to `apple_vpio_bridge` would corrupt capture interpretation.

## Remaining validation

Live VPIO/raw startup, microphone permissions, channel routing, cancellation, speaker/microphone interaction, echo suppression, playback timing, and long-running latency have not been tested during reconstruction. The derivative follows the recovered capture approach but does not restore unobserved channel-selection or audio-device logic that may have existed in the missing original helper. The SoundDevice/legacy route remains available as an alternative.

The unchanged Python native playback wrapper returns after writing a WAV rather than waiting for playback completion. Consequently its `tts_playing` flag and playback guard can stop reflecting native playback before speech finishes. The helper bounds queued playback but does not add an acknowledgment channel to the saved Python protocol. Full completion/echo-guard coordination requires an explicit Python/protocol change and live validation.
