from chat.chat import ChatPart
from tts.tts_api import TTSApi, base64_to_wav
from yolo.yolo import YoloV5

# 加载yolo part
input_image_path = ""
yolo_part = YoloV5()
yolo_result = yolo_part.detect(input_image_path)

# 加载chat part
chat_part = ChatPart()
chat_part.load_model()
chat_part.write_input("请根据以下yolo输出结果描述图片内容："+yolo_result)
text = chat_part.check_input()
output_text = chat_part.generate(text)
chat_part.unload()

# 加载tts part
ak = ""
sk = ""
tts_part = TTSApi(ak, sk)
response = tts_part.generate_tts(text)
base64_to_wav(response.result.data, "out.wav")
