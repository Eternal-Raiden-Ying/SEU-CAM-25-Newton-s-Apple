import struct

with open('file14.wav', 'rb') as f:
    data = f.read()

# 保留音频数据部分
audio_data = data[44:]  # 假设后面是有效数据

# 构造新 header
chunk_size = 36 + len(audio_data)
byte_rate = 8000 * 1 * 2  # SampleRate * Channels * BytesPerSample
block_align = 1 * 2
subchunk2_size = len(audio_data)

header = b'RIFF' + struct.pack('<I', chunk_size) + b'WAVE'
header += b'fmt ' + struct.pack('<IHHIIHH',
                                16,     # Subchunk1Size
                                1,      # PCM
                                1,      # Mono
                                8000,   # SampleRate
                                byte_rate,
                                block_align,
                                16)     # BitsPerSample
header += b'data' + struct.pack('<I', subchunk2_size)

with open('fixed.wav', 'wb') as f:
    f.write(header + audio_data)