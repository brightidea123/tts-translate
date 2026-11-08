import Foundation
import AVFoundation

// Binary stdout protocol per frame:
// magic: 4 bytes "APCM"
// timestamp_ns: UInt64 little-endian, monotonic host clock
// sample_rate: UInt32 little-endian, always 16000
// samples: UInt32 little-endian, number of Int16 mono samples
// payload: samples * 2 bytes, Int16 little-endian PCM

let targetRate: Double = 16_000
let targetChannels: AVAudioChannelCount = 1
let frameSamples = 320 // 20 ms @ 16 kHz
let mode = ProcessInfo.processInfo.environment["APPLE_VPIO_MODE"] ?? "vpio"
let agc = ProcessInfo.processInfo.environment["APPLE_VPIO_AGC"] ?? "on"
let bypass = ProcessInfo.processInfo.environment["APPLE_VPIO_BYPASS"] ?? "off"
let verbose = ProcessInfo.processInfo.environment["APPLE_VPIO_VERBOSE"] == "1"

func le32(_ x: UInt32) -> [UInt8] {
    let v = x.littleEndian
    return withUnsafeBytes(of: v) { Array($0) }
}

func le64(_ x: UInt64) -> [UInt8] {
    let v = x.littleEndian
    return withUnsafeBytes(of: v) { Array($0) }
}

func hostTimeNs() -> UInt64 {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let now = mach_absolute_time()
    return now * UInt64(info.numer) / UInt64(info.denom)
}

final class StdoutFramer {
    private let q = DispatchQueue(label: "stdout-framer")
    private var carry = [Int16]()

    func push(_ samples: [Int16], timestampNs: UInt64) {
        q.async {
            self.carry.append(contentsOf: samples)
            while self.carry.count >= frameSamples {
                let chunk = Array(self.carry.prefix(frameSamples))
                self.carry.removeFirst(frameSamples)
                self.writeFrame(chunk, timestampNs: timestampNs)
            }
        }
    }

    private func writeFrame(_ samples: [Int16], timestampNs: UInt64) {
        var out = Data()
        out.append(contentsOf: [0x41, 0x50, 0x43, 0x4D]) // APCM
        out.append(contentsOf: le64(timestampNs))
        out.append(contentsOf: le32(UInt32(targetRate)))
        out.append(contentsOf: le32(UInt32(samples.count)))
        samples.withUnsafeBufferPointer { ptr in
            let bytes = UnsafeRawBufferPointer(ptr)
            out.append(contentsOf: bytes)
        }
        FileHandle.standardOutput.write(out)
    }
}

func int16FromFloatBuffer(_ buffer: AVAudioPCMBuffer) -> [Int16] {
    guard let channel = buffer.floatChannelData?[0] else { return [] }
    let n = Int(buffer.frameLength)
    var out = [Int16](repeating: 0, count: n)
    for i in 0..<n {
        let x = max(-1.0, min(1.0, channel[i]))
        out[i] = Int16(x * 32767.0)
    }
    return out
}

let engine = AVAudioEngine()
let input = engine.inputNode
let output = engine.outputNode

if mode == "vpio" {
    do {
        try input.setVoiceProcessingEnabled(true)
        try output.setVoiceProcessingEnabled(true)
        if #available(macOS 12.0, *) {
            input.isVoiceProcessingAGCEnabled = (agc == "on")
            input.isVoiceProcessingInputMuted = false
            input.isVoiceProcessingBypassed = (bypass == "on")
        }
        fputs("apple-vpio-bridge ready mode=vpio agc=\(agc) bypass=\(bypass)\n", stderr)
    } catch {
        fputs("apple-vpio failed, continuing raw: \(error)\n", stderr)
    }
} else {
    fputs("apple-vpio-bridge ready mode=raw\n", stderr)
}

let inputFormat = input.outputFormat(forBus: 0)
guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                       sampleRate: targetRate,
                                       channels: targetChannels,
                                       interleaved: false) else {
    fputs("failed to create target audio format\n", stderr)
    exit(2)
}

guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
    fputs("failed to create AVAudioConverter from \(inputFormat) to \(targetFormat)\n", stderr)
    exit(2)
}

let framer = StdoutFramer()

input.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { buffer, _ in
    let ts = hostTimeNs()
    let ratio = targetRate / inputFormat.sampleRate
    let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 32)
    guard let outBuf = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }

    var consumed = false
    let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
        if consumed {
            outStatus.pointee = .noDataNow
            return nil
        }
        consumed = true
        outStatus.pointee = .haveData
        return buffer
    }

    do {
        try converter.convert(to: outBuf, error: nil, withInputFrom: inputBlock)
        let pcm = int16FromFloatBuffer(outBuf)
        if !pcm.isEmpty { framer.push(pcm, timestampNs: ts) }
    } catch {
        if verbose { fputs("convert error: \(error)\n", stderr) }
    }
}

do {
    try engine.start()
} catch {
    fputs("engine start failed: \(error)\n", stderr)
    exit(3)
}

RunLoop.main.run()
