from chat.chat import ChatPart

chat_part = ChatPart()
chat_part.load_model()
chat_part.write_input('你好')
text = chat_part.check_input()
chat_part.generate(text)