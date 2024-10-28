class DataPipeline:

    def __init___(self):
        self.chat_history = {}
    
    def get_inputs(self, session_id):

        if self.chat_history[session_id]:
            for msg in self.chat_history[session_id]:
                if msg.role == "user":
                    print("\nUser Input:" + msg.content)

    def get_outputs(self, session_id):

        if self.chat_history[session_id]:
            for msg in self.chat_history[session_id]:
                if msg.role == "system":
                    print("\nSystem Output:" + msg.content)
    
    def get_full_chat_history(self, session_id, output_file):

        file = open(output_file, "w") 

        chat = self.chat_history[session_id]

        file.write("CHAT HISTORY " + session_id + "\n")
        for msg in chat:
            if msg.role == "user":
                file.write("USER: " + msg.content + "\n")
            
            else:
                file.write("SYSTEM: " + msg.content + "\n")

        file.close() 
                    

    def add_message(self, role, content, session_id):
        msg = Message(role, content)
        self.chat_history[session_id].append(msg)


class Message:
    _role: str
    _content: str

    def __init__(self, role, content):
        self._role = role
        self._content = content
