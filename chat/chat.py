import json
import os
import time

from filelock import FileLock
from ais_bench.infer.interface import InferSession
from utils import content_generate, preprocess, postprocess, generate_onnx_input


class ChatPart:
    def __init__(self):
        self.temp_path = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path, exist_ok=True)
        self.input_filepath = os.path.join(self.temp_path, 'input.txt')
        self.output_filepath = os.path.join(self.temp_path, 'output.json')
        lock_filepath = os.path.join(self.temp_path, 'lock.txt')
        self.lock = FileLock(lock_filepath, timeout=5)
        self.encoder = None
        self.first_decoder = None
        self.decoder_iter = None
        self.tokenizer = None
        self.dummy_decoder_input_ids = None
        self.logits_processor = None
        self.logits_warper = None
        self.stopping_criteria = None
        self.eos_token_id = None
        self.record = None

    # 加载模型
    def load_model(self):
        self.encoder = InferSession(0, './models/encoder.om')
        import numpy as np
        import onnxruntime
        from transformers import T5Tokenizer
        from transformers.generation import LogitsProcessorList, NoRepeatNGramLogitsProcessor, TemperatureLogitsWarper, \
            TopKLogitsWarper, StoppingCriteriaList, MaxLengthCriteria
        print('[INFO]The encoder has been initialized. Initializing the first decoder in progress.')
        self.first_decoder = onnxruntime.InferenceSession('../models/decoder_first_sim_quant.onnx')
        print('[INFO]The first decoder has been initialized. Initializing the second decoder in progress.')
        self.decoder_iter = onnxruntime.InferenceSession('../models/decoder_iter_sim_quant.onnx')
        self.tokenizer = T5Tokenizer.from_pretrained("./tokenizer")
        self.dummy_decoder_input_ids = np.array([[0]], dtype=np.int64)
        self.logits_processor = LogitsProcessorList([NoRepeatNGramLogitsProcessor(3)])
        self.logits_warper = LogitsProcessorList(
            [TemperatureLogitsWarper(0.7), TopKLogitsWarper(filter_value=float('-inf'), top_k=50)])
        self.stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(512)])
        self.eos_token_id = [1]
        self.record = []
        print('[INFO]init finished')

    # 写入输入文件
    def write_input(self, text):
        with self.lock:
            with open(self.input_filepath, 'w', encoding='utf-8') as f:
                f.write(text)

    # 检查是否有待处理的输入文件
    def check_input(self):
        with self.lock:
            try:
                with open(self.input_filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                # 删除或清空文件内容
                os.remove(self.input_filepath)
                return text
            except FileNotFoundError:
                return None

    # 将输出结果写入文件
    def write_output_json(self, data):
        with self.lock:
            while os.path.exists(self.output_filepath):
                time.sleep(0.1)
            with open(self.output_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    # 生成输出结果
    def generate(self, input_text, output=False):
        import torch
        if input_text == 'clear' and output:
            print('record clear')
            self.record.clear()
            data = {
                "code": 200,
                "data": {
                    "isEnd": False,
                    "message": '聊天记录已清空'
                }
            }
            self.write_output_json(data)
            return

        # 生成附带上下文的模型输入
        content = content_generate(self.record, input_text)

        # 对输入文本进行预处理，生成token和attention_mask
        inputs = self.tokenizer(text=[preprocess(content)], truncation=True, padding='max_length', max_length=768,
                                return_tensors="np")

        encoder_input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # 使用encoder模型生成encoder_hidden_states
        encoder_hidden_states = self.encoder.infer([encoder_input_ids, attention_mask])[0]

        print('autoregression start')
        first_loop = True
        decoder_input_ids = self.dummy_decoder_input_ids

        input_ids = torch.tensor(self.dummy_decoder_input_ids)
        unfinished_sequences = torch.tensor([1])
        while True:
            if first_loop:
                outputs = self.first_decoder.run(None, {'decoder_input_ids': decoder_input_ids,
                                                        'hidden_states': encoder_hidden_states,
                                                        'attention_mask': attention_mask})
                first_loop = False
            else:
                onnx_input = generate_onnx_input(decoder_input_ids, attention_mask, past_key_values)
                outputs = self.decoder_iter.run(None, onnx_input)
            logits = torch.tensor(outputs[0])
            past_key_values = outputs[1:]
            next_token_logits = logits[:, -1, :]

            next_token_scores = self.logits_processor(input_ids, next_token_logits)
            next_token_scores = self.logits_warper(input_ids, next_token_scores)

            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            message = postprocess(self.tokenizer.batch_decode(next_tokens, skip_special_tokens=True)[0])
            if message == ' ' or message == '':
                message = '&nbsp'
            elif message == '\n':
                message = '<br />'

            data = {
                "code": 200,
                "data": {
                    "isEnd": False,
                    "message": message
                }
            }

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            decoder_input_ids = input_ids[:, -1:].numpy()

            # 判断是否结束
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in self.eos_token_id)).long())
            if unfinished_sequences.max() == 0 or self.stopping_criteria(input_ids, None):
                data['data']['isEnd'] = True
                if output:
                    self.write_output_json(data)

                break
            else:
                if output:
                    self.write_output_json(data)

        out_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        out_text = postprocess(out_text)

        self.record.append([input_text, out_text])
        if self.tokenizer(text=[preprocess(content_generate(self.record, ''))], truncation=True, padding=True,
                          max_length=768,
                          return_tensors="np")['attention_mask'].shape[1] > 256:
            print('record clear')
            self.record.clear()

    # 卸载模型
    def unload(self):
        del self.encoder
        del self.first_decoder
        del self.decoder_iter
        del self.tokenizer
        del self.dummy_decoder_input_ids
        del self.logits_processor
        del self.logits_warper
        del self.stopping_criteria
        del self.eos_token_id
        del self.record
        print('[INFO]model unloaded')
