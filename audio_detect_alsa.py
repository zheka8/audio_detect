#!/usr/bin/python3
import alsaaudio
import wave

# Set up audio input
input_device = 'hw:2,0'
input_channels = 1
input_rate = 44100
input_format = alsaaudio.PCM_FORMAT_S16_LE
input_period_size = 1024

input_stream = alsaaudio.PCM(
    alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, input_device
)

input_stream.setchannels(input_channels)
input_stream.setrate(input_rate)
input_stream.setformat(input_format)
input_stream.setperiodsize(input_period_size)

# Record audio
frames = []
for i in range(0, int(input_rate / input_period_size * 10)):
    # Read data from the input stream
    data = input_stream.read()
    #print(data)
    #print(type(data))
    frames.append(data[1])

# Save to WAV file
output_file = wave.open('output.wav', 'wb')
output_file.setnchannels(input_channels)
output_file.setsampwidth(2) #16 bit sample width
output_file.setframerate(input_rate)
output_file.writeframes(b''.join(frames))
output_file.close()

# Clean up
input_stream.close()
