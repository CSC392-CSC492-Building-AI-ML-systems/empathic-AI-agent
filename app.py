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

# for now, store chat history as a dict
chat_histories = {}

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
        self._agent1_executor = create_react_agent(
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
        self._agent2_executor = create_react_agent(
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

        if session_id not in chat_histories:
            chat_histories[session_id] = []

        chat_histories[session_id].append({"role": "user", "content": message})

        messages = [HumanMessage(content=msg["content"]) for msg in chat_histories[session_id]]

        response1 = self._agent1_executor.invoke(
            {"messages": messages},
            config
        )
        clarification_result = response1["messages"][-1].content

        # Append the agent1's response to chat history
        chat_histories[session_id].append({"role": "system", "content": clarification_result})

        combined_message = message + clarification_result

        response2 = self._agent2_executor.invoke(
            {"messages": [HumanMessage(content=combined_message)]},
            config
        )

        final_response = response2['messages'][-1].content

        chat_histories[session_id].append({"role": "system", "content": final_response})

        return final_response

chat_app = ChatApp()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    message = data.get('message')
    session_id = data.get('session_id')

    if not session_id:
        session_id = generate_random_session_id()

    try:
        response = chat_app.submit_message(message, session_id)
        return jsonify({'response': response, 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/new_chat', methods=['POST'])
def new_chat():
    session_id = generate_random_session_id()
    return jsonify({'session_id': session_id})

if __name__ == '__main__':
    app.run(debug=True)
    print(chat_histories)
