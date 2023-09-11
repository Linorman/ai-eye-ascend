from .models.asr_model import WeNetASR


class ASRPart:
    def __init__(self):
        self.wenet_model_path = 'models/offline_encoder.om'
        self.wenet_vocab_path = 'models/vocab.txt'

    def run_asr(self, input_path):
        """语音识别模型推理，并返回识别文本。"""
        print('Loading models...')
        model = WeNetASR(self.wenet_model_path, self.wenet_vocab_path)
        print('asr ready')
        if input_path is None:
            print('input is None')
            return
        input_wav_file = input_path.get()
        text = model.transcribe(input_wav_file)
        print('asr result: ', text)
        return text
