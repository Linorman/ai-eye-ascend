from tts.tts_api import TTSApi

ak = "<YOUR AK>"
sk = "<YOUR SK>"
text = "你好啊"

tts_client = TTSApi(ak, sk)
response = tts_client.generate_tts(text)

print(response)
