import requests


class TTSApi:
    def __init__(self, project_id, auth):
        self.url = "https://sis-ext.cn-east-3.myhuaweicloud.com/v1/" + project_id + "/tts"
        self.headers = {
            'Authorization': auth
        }

    def generate_speech(self, input, audio_format='wav', sample_rate='16000', property='chinese_huaxiaoliang_common',
                        speed=0, pitch=0, volume=50):
        payload = {
            "text": input,
            "config": {
                "audio_format": audio_format,
                "sample_rate": sample_rate,
                "property": property,
                "speed": speed,
                "pitch": pitch,
                "volume": volume
            }
        }

        response = requests.request("POST", self.url, headers=self.headers, data=payload)

        if response.status_code == 200:
            return response.content
        else:
            raise Exception("Speech generation failed. Status code: {}".format(response.status_code))
