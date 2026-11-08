// Reconstructed raw-duplex derivative, 2026-11-06.
// Capture/conversion adapted from the recovered apple_vpio_timestamp_bridge.swift.
// This is NOT the recovered original missing apple_vpio_bridge.swift.
// See docs/NATIVE_BRIDGE_RECOVERY.md for provenance, protocol, and validation.
import Foundation
import AVFoundation
import Darwin

private let targetRate: Double = 16_000
private let frameSamples = 320
private let maximumWAVBytes = 64 * 1024 * 1024

private func diagnostic(_ text: String) {
    FileHandle.standardError.write(Data((text + "\n").utf8))
}

private enum BridgeError: Error, CustomStringConvertible {
    case invalid(String)
    var description: String {
        switch self { case .invalid(let message): return message }
    }
}

// The same encoder/framer is used by capture and by --self-test.
private func encodePCM16(_ samples: ArraySlice<Int16>) -> Data {
    var result = Data(capacity: samples.count * 2)
    for sample in samples {
        var value = sample.littleEndian
        withUnsafeBytes(of: &value) { result.append(contentsOf: $0) }
    }
    return result
}

private final class RawPCMFramer {
    private var carry: [Int16] = []
    private let emit: (Data) throws -> Void

    init(emit: @escaping (Data) throws -> Void) { self.emit = emit }

    func push(_ samples: [Int16]) throws {
        carry.append(contentsOf: samples)
        var consumed = 0
        while carry.count - consumed >= frameSamples {
            try emit(encodePCM16(carry[consumed..<(consumed + frameSamples)]))
            consumed += frameSamples
        }
        if consumed > 0 { carry.removeFirst(consumed) }
    }
}

// Accept arbitrary pipe fragmentation and consecutive length-prefixed WAV messages.
private final class WAVStreamParser {
    private var pending = Data()
    private var expected: Int?
    private let receive: (Data) throws -> Void

    init(receive: @escaping (Data) throws -> Void) { self.receive = receive }

    func feed(_ bytes: Data) throws {
        pending.append(bytes)
        while true {
            if expected == nil {
                guard pending.count >= 4 else { return }
                let header = Array(pending.prefix(4))
                let length = Int(header[0]) | Int(header[1]) << 8
                    | Int(header[2]) << 16 | Int(header[3]) << 24
                guard length >= 12 && length <= maximumWAVBytes else {
                    throw BridgeError.invalid("invalid playback WAV length: \(length)")
                }
                pending.removeFirst(4)
                expected = length
            }
            guard let length = expected, pending.count >= length else { return }
            let wav = Data(pending.prefix(length))
            pending.removeFirst(length)
            expected = nil
            guard wav.prefix(4) == Data("RIFF".utf8),
                  wav.subdata(in: 8..<12) == Data("WAVE".utf8) else {
                throw BridgeError.invalid("playback message is not a RIFF/WAVE file")
            }
            try receive(wav)
        }
    }

    func finish() throws {
        guard pending.isEmpty && expected == nil else {
            throw BridgeError.invalid("truncated playback message at stdin EOF")
        }
    }
}

// AVAudioFile validates the actual WAV content. The temporary file is removed
// immediately after decoding; AVAudioPlayerNode later receives an in-memory buffer.
private func decodeWAV(_ wav: Data) throws -> AVAudioPCMBuffer {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("tts-bridge-\(UUID().uuidString).wav")
    defer { try? FileManager.default.removeItem(at: url) }
    try wav.write(to: url)
    let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
    guard file.length > 0, file.length <= Int64(maximumWAVBytes / 2),
          let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                       frameCapacity: AVAudioFrameCount(file.length)) else {
        throw BridgeError.invalid("empty or oversized decoded playback WAV")
    }
    try file.read(into: buffer)
    return buffer
}

private func resamplePlayback(_ source: AVAudioPCMBuffer,
                              to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
    guard let converter = AVAudioConverter(from: source.format, to: format) else {
        throw BridgeError.invalid("cannot convert playback format")
    }
    let capacity = AVAudioFrameCount(ceil(Double(source.frameLength)
        * format.sampleRate / source.format.sampleRate) + 64)
    guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
        throw BridgeError.invalid("cannot allocate playback buffer")
    }
    var supplied = false
    var conversionError: NSError?
    let status = converter.convert(to: output, error: &conversionError) { _, state in
        if supplied { state.pointee = .endOfStream; return nil }
        supplied = true
        state.pointee = .haveData
        return source
    }
    if status == .error {
        throw conversionError ?? BridgeError.invalid("playback conversion failed") as NSError
    }
    guard output.frameLength > 0 else {
        throw BridgeError.invalid("playback conversion produced no frames")
    }
    return output
}

private func floatPCM16(_ buffer: AVAudioPCMBuffer) -> [Int16] {
    guard let channel = buffer.floatChannelData?[0] else { return [] }
    return (0..<Int(buffer.frameLength)).map { i in
        let value = channel[i].isFinite ? max(-1, min(1, channel[i])) : 0
        return Int16(value * 32767)
    }
}

private func runSelfTest() throws {
    func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
        if !condition() { throw BridgeError.invalid("self-test: " + message) }
    }
    func u32(_ value: UInt32) -> Data {
        var little = value.littleEndian
        return withUnsafeBytes(of: &little) { Data($0) }
    }
    func mustReject(_ message: String, _ operation: () throws -> Void) throws {
        do { try operation() } catch { return }
        throw BridgeError.invalid("self-test: accepted " + message)
    }

    // Independently specified edge samples and expected byte order, crossing two
    // 20 ms output boundaries with an unfinished tail and subsequent completion.
    let edge: [Int16] = [-32768, -1, 0, 1, 32767]
    let samples = Array(repeating: edge, count: 128).flatMap { $0 }
    let expectedEdge = Data([0, 128, 255, 255, 0, 0, 1, 0, 255, 127])
    var frames: [Data] = []
    let framer = RawPCMFramer { frames.append($0) }
    try framer.push(Array(samples.prefix(319)))
    try require(frames.isEmpty, "partial capture frame emitted early")
    try framer.push(Array(samples.dropFirst(319).prefix(17)))
    try require(frames.count == 1 && frames[0].count == 640, "first raw frame boundary")
    try framer.push(Array(samples.dropFirst(336)))
    try require(frames.count == 2, "carry samples lost across callbacks")
    try require(frames.reduce(Data(), +) == (0..<128).reduce(into: Data()) { data, _ in
        data.append(expectedEdge)
    }, "capture bytes were framed, reordered, or not little-endian")

    // Known PCM RIFF fixture, parsed from fragmented stdin and decoded by the same
    // AVAudioFile path used for real playback, without creating an audio engine.
    let pcm = Data([0, 128, 0, 192, 0, 0, 0, 64, 255, 127])
    var wav = Data("RIFF".utf8); wav.append(u32(UInt32(36 + pcm.count)))
    wav.append(Data("WAVEfmt ".utf8)); wav.append(u32(16))
    wav.append(Data([1, 0, 1, 0])); wav.append(u32(16_000)); wav.append(u32(32_000))
    wav.append(Data([2, 0, 16, 0])); wav.append(Data("data".utf8))
    wav.append(u32(UInt32(pcm.count))); wav.append(pcm)
    var received: [Data] = []
    let parser = WAVStreamParser { received.append($0) }
    var stream = u32(UInt32(wav.count)); stream.append(wav)
    stream.append(u32(UInt32(wav.count))); stream.append(wav)
    for byte in stream { try parser.feed(Data([byte])) }
    try parser.finish()
    try require(received == [wav, wav], "fragmented/consecutive stdin messages corrupted")
    let decoded = try decodeWAV(received[0])
    try require(decoded.frameLength == 5 && decoded.format.sampleRate == 16_000
        && decoded.format.channelCount == 1, "decoded WAV shape")
    guard let channel = decoded.floatChannelData?[0] else {
        throw BridgeError.invalid("self-test: missing decoded samples")
    }
    let expected: [Float] = [-1, -0.5, 0, 0.5, 32767.0 / 32768.0]
    try require((0..<5).allSatisfy { abs(channel[$0] - expected[$0]) < 0.0001 },
                "actual WAV decoder changed known samples")
    let converted = try resamplePlayback(decoded, to: decoded.format)
    try require(converted.frameLength == 5, "actual playback converter lost frames")
    guard let stereoFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
        sampleRate: 48_000, channels: 2, interleaved: false) else {
        throw BridgeError.invalid("self-test: missing stereo test format")
    }
    let stereo = try resamplePlayback(decoded, to: stereoFormat)
    try require(stereo.frameLength > 0 && stereo.format.channelCount == 2
        && stereo.format.sampleRate == 48_000, "actual playback rate/channel conversion failed")
    try mustReject("oversized WAV") {
        try WAVStreamParser { _ in }.feed(u32(UInt32(maximumWAVBytes + 1)))
    }
    try mustReject("truncated WAV") {
        let truncated = WAVStreamParser { _ in }
        try truncated.feed(stream.prefix(9)); try truncated.finish()
    }
    try mustReject("non-WAV payload") {
        let invalid = WAVStreamParser { _ in }
        var bytes = u32(12); bytes.append(Data(repeating: 0, count: 12))
        try invalid.feed(bytes)
    }
    diagnostic("self-test OK: raw PCM boundaries/endianness, fragmented WAV stream, real WAV decode/conversion, invalid/truncated input")
}

private final class DuplexBridge {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let captureQueue = DispatchQueue(label: "tts-bridge-capture-output")
    // Bound work retained while stdout is backpressured; never block the tap.
    private let captureSlots = DispatchSemaphore(value: 4)
    private let playbackQueue = DispatchQueue(label: "tts-bridge-playback")
    private let playbackSlots = DispatchSemaphore(value: 2)
    private let stateLock = NSLock()
    private var capturedFrames = 0
    private var signals: [DispatchSourceSignal] = []
    private var tapInstalled = false
    private let env = ProcessInfo.processInfo.environment
    private var debug: Bool { env["APPLE_VPIO_DEBUG"] == "1" }
    private var verbose: Bool { debug || env["APPLE_VPIO_VERBOSE"] == "1" }

    func start() throws {
        let input = engine.inputNode
        let output = engine.outputNode
        let requested = env["APPLE_VPIO_MODE"] ?? "vpio"
        guard ["raw", "vpio"].contains(requested) else {
            throw BridgeError.invalid("APPLE_VPIO_MODE must be raw or vpio")
        }
        var actualMode = requested
        if requested == "vpio" {
            do {
                try input.setVoiceProcessingEnabled(true)
                try output.setVoiceProcessingEnabled(true)
                if #available(macOS 12.0, *) {
                    input.isVoiceProcessingAGCEnabled = env["APPLE_VPIO_AGC"] != "off"
                    input.isVoiceProcessingInputMuted = false
                    input.isVoiceProcessingBypassed = env["APPLE_VPIO_BYPASS"] == "on"
                }
            } catch {
                try? input.setVoiceProcessingEnabled(false)
                try? output.setVoiceProcessingEnabled(false)
                actualMode = "raw"
                diagnostic("apple-vpio failed, continuing raw: \(error)")
            }
        }
        let inputFormat = input.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0,
              let captureFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                  sampleRate: targetRate, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inputFormat, to: captureFormat) else {
            throw BridgeError.invalid("no valid microphone input/conversion format")
        }
        engine.attach(player)
        let playbackFormat = engine.mainMixerNode.outputFormat(forBus: 0)
        guard playbackFormat.sampleRate > 0, playbackFormat.channelCount > 0 else {
            throw BridgeError.invalid("no valid playback format")
        }
        engine.connect(player, to: engine.mainMixerNode, format: playbackFormat)
        let framer = RawPCMFramer { try FileHandle.standardOutput.write(contentsOf: $0) }
        input.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            let capacity = AVAudioFrameCount(ceil(Double(buffer.frameLength)
                * targetRate / inputFormat.sampleRate) + 64)
            guard let converted = AVAudioPCMBuffer(pcmFormat: captureFormat, frameCapacity: capacity) else { return }
            var supplied = false
            var conversionError: NSError?
            let status = converter.convert(to: converted, error: &conversionError) { _, state in
                if supplied { state.pointee = .noDataNow; return nil }
                supplied = true; state.pointee = .haveData; return buffer
            }
            if status == .error {
                if self.verbose { diagnostic("convert error: \(String(describing: conversionError))") }
                return
            }
            let pcm = floatPCM16(converted)
            guard !pcm.isEmpty else { return }
            self.stateLock.lock(); self.capturedFrames += 1; self.stateLock.unlock()
            guard self.captureSlots.wait(timeout: .now()) == .success else {
                if self.debug { diagnostic("capture output backpressure: dropped audio chunk") }
                return
            }
            self.captureQueue.async {
                defer { self.captureSlots.signal() }
                do { try framer.push(pcm) }
                catch { diagnostic("capture output failed: \(error)"); exit(4) }
            }
        }
        tapInstalled = true
        engine.prepare()
        try engine.start()
        player.play()
        diagnostic("apple-vpio-bridge ready mode=\(actualMode) agc=\(env["APPLE_VPIO_AGC"] ?? "on") bypass=\(env["APPLE_VPIO_BYPASS"] ?? "off")")
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            guard let self = self else { return }
            self.stateLock.lock(); let count = self.capturedFrames; self.stateLock.unlock()
            if count == 0 { diagnostic("no input frames received within 2s") }
        }
        playbackQueue.async { [weak self] in
            guard let self = self else { return }
            let parser = WAVStreamParser { wav in
                let decoded = try decodeWAV(wav)
                let converted = try resamplePlayback(decoded, to: playbackFormat)
                self.playbackSlots.wait()
                self.player.scheduleBuffer(converted, completionCallbackType: .dataPlayedBack) { _ in
                    self.playbackSlots.signal()
                }
                if self.debug { diagnostic("playback bytes=\(wav.count)") }
            }
            do {
                while true {
                    let data = FileHandle.standardInput.availableData
                    if data.isEmpty { break }
                    try parser.feed(data)
                }
                try parser.finish()
                DispatchQueue.main.async { self.stop(); exit(0) }
            } catch {
                diagnostic("playback input failed: \(error)")
                DispatchQueue.main.async { self.stop(); exit(5) }
            }
        }
        for number in [SIGINT, SIGTERM] {
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
            source.setEventHandler { [weak self] in self?.stop(); exit(0) }
            source.resume(); signals.append(source)
        }
    }

    func stop() {
        if tapInstalled { engine.inputNode.removeTap(onBus: 0); tapInstalled = false }
        player.stop()
        engine.stop()
    }
}

// This branch precedes AVAudioEngine creation and never requests microphone access.
if CommandLine.arguments.dropFirst().contains("--self-test") {
    do { try runSelfTest(); exit(0) }
    catch { diagnostic("\(error)"); exit(1) }
}
if CommandLine.arguments.dropFirst().contains("--help") {
    diagnostic("apple_vpio_bridge [--self-test] | raw PCM stdout; length-prefixed WAV stdin; APPLE_VPIO_MODE=raw|vpio")
    exit(0)
}
signal(SIGPIPE, SIG_IGN)
private let bridge = DuplexBridge()
do { try bridge.start() }
catch { diagnostic("engine start failed: \(error)"); bridge.stop(); exit(3) }
RunLoop.main.run()
