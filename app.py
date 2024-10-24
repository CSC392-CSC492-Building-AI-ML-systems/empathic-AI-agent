from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from utilities import generate_random_session_id

app = Flask(__name__)

class ChatApp:
    _model: ChatOpenAI
    _tools: list[BaseTool]
    _memory: MemorySaver
    _agent1_executor: CompiledGraph
    _agent2_executor: CompiledGraph

    def __init__(self):
        load_dotenv()
        self._model = ChatOpenAI(model="gpt-4")
        self._tools = [TavilySearchResults(max_results=2)]
        self._memory = MemorySaver()
        self._agent1_executer = create_react_agent(
            self._model, 
            self._tools, 
            state_modifier='''
            Analyze the user's request and determine if it requires clarification due to ambiguity.
            If clarification is needed:
            1. Respond with "CLARIFICATION_NEEDED: " followed by the clarifying question.
            2. Example: "CLARIFICATION_NEEDED: Are you asking about servers in the context of computers or food service?"

            If no clarification is needed:
            1. Respond with "NO_CLARIFICATION_NEEDED"

            Always provide only one of these two responses, with no additional text.
            ''',
            checkpointer=self._memory
        )
        self._agent2_executer = create_react_agent(
            self._model, 
            self._tools, 
            state_modifier='''
            You are an assistant that provides information based on the user's request.
            You will receive input in one of two formats:

            1. A user query followed by "NO_CLARIFICATION_NEEDED"
               In this case, provide a detailed response to the user's query.

            2. A user query followed by "CLARIFICATION_NEEDED: [question]"
               In this case, ask the clarifying question provided.

            Maintain a conversational tone and ensure your response is appropriate to the input received.
            ''',
            checkpointer=self._memory
        )

    def submit_message(self, message: str, session_id: str) -> str:
        config = {"configurable": {"thread_id": session_id}}

        response1 = self._agent1_executer.invoke(
            {"messages": [HumanMessage(content=message)]}, 
            config
        )
        clarification_result = response1["messages"][-1].content

        response2 = self._agent2_executer.invoke(
            {"messages": [HumanMessage(content=message + clarification_result)]}, 
            config
        )

        return response2['messages'][-1].content

chat_app = ChatApp()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    message = request.json.get('message')
    session_id = request.json.get('session_id', generate_random_session_id())
    
    try:
        response = chat_app.submit_message(message, session_id)
        return jsonify({'response': response, 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Chat Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        #chat-container {
            height: 400px;
            border: 1px solid #ccc;
            padding: 20px;
            overflow-y: auto;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
        }
        .system-message {
            background-color: #f5f5f5;
            margin-right: 20%;
        }
        #input-container {
            display: flex;
            gap: 10px;
        }
        #message-input {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        button {
            padding: 10px 20px;
            background-color: #2196f3;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:disabled {
            background-color: #ccc;
        }
        .error {
            color: red;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Chat Interface</h1>
    <div id="chat-container"></div>
    <div id="input-container">
        <input type="text" id="message-input" placeholder="Type your message...">
        <button onclick="sendMessage()" id="send-button">Send</button>
    </div>
    <div id="error" class="error"></div>

    <script>
        let sessionId = null;
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const errorDiv = document.getElementById('error');

        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Disable input and button while processing
            messageInput.disabled = true;
            sendButton.disabled = true;

            // Add user message to chat
            addMessage(message, 'user');
            messageInput.value = '';

            try {
                const response = await fetch('/submit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        session_id: sessionId
                    })
                });

                const data = await response.json();
                
                if (response.ok) {
                    sessionId = data.session_id;
                    addMessage(data.response, 'system');
                    errorDiv.textContent = '';
                } else {
                    throw new Error(data.error || 'An error occurred');
                }
            } catch (error) {
                errorDiv.textContent = error.message;
                console.error('Error:', error);
            } finally {
                // Re-enable input and button
                messageInput.disabled = false;
                sendButton.disabled = false;
                messageInput.focus();
            }
        }

        function addMessage(message, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.textContent = message;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>
'''

import os
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write(html_template)

if __name__ == '__main__':
    app.run(debug=True)