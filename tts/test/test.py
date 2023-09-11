from tts.asr import ASRPart

input_wav_file = 'test.wav'
asr_part = ASRPart()
text = asr_part.run_asr(input_wav_file)
print(text)
