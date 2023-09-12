from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdksis.v1.region.sis_region import SisRegion
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdksis.v1 import *

import base64
import wave


def base64_to_wav(base64_data, output_path):
    audio_bytes = base64.b64decode(base64_data)
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_bytes)


class TTSApi:
    def __init__(self, ak, sk, region="cn-east-3"):
        credentials = BasicCredentials(ak, sk)

        self.client = SisClient.new_builder() \
            .with_credentials(credentials) \
            .with_region(SisRegion.value_of(region)) \
            .build()

    def generate_tts(self, text, audio_format="wav", sample_rate="16000", property="chinese_huaxiaoliang_common",
                     speed=0, pitch=0, volume=50):
        try:
            request = RunTtsRequest()
            configbody = TtsConfig(
                audio_format=audio_format,
                sample_rate=sample_rate,
                _property=property,
                speed=speed,
                pitch=pitch,
                volume=volume
            )
            request.body = PostCustomTTSReq(
                config=configbody,
                text=text
            )
            response = self.client.run_tts(request)
            return response
        except exceptions.ClientRequestException as e:
            print(e.status_code)
            print(e.request_id)
            print(e.error_code)
            print(e.error_msg)
