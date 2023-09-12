from tts.tts_api import TTSApi, base64_to_wav

ak = ""
sk = ""
text = "你好啊"

tts_client = TTSApi(ak, sk)
response = tts_client.generate_tts(text)
base64_to_wav(response.result.data, "test.wav")
print(response)
