# app.py

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
from data_pipeline import DataPipeline
from utilities import generate_random_session_id
from datetime import datetime

app = Flask(__name__)
load_dotenv()  

LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# store chat histories
chat_histories = {}

prompt1 = '''
Analyze the user's request and determine if it requires clarification 
due to ambiguity.
If clarification is needed:
1. Respond with "CLARIFICATION_NEEDED: " followed by the clarifying question.
2. Example: "CLARIFICATION_NEEDED: Are you asking about servers in the context 
                of computers or food service?"

If no clarification is needed:
1. Respond with "NO_CLARIFICATION_NEEDED"

Always provide only one of these two responses, with no additional text.
'''

prompt2 = '''
You are an assistant that provides information based on the user's request.
You will receive input in one of two formats:

1. A user query followed by "NO_CLARIFICATION_NEEDED"
In this case, provide a detailed response to the user's query.

2. A user query followed by "CLARIFICATION_NEEDED: [question]"
In this case, ask the clarifying question provided.

Maintain a conversational tone and ensure your response is appropriate 
to the input received.

Format the output as "QUESTION_AGENT_OUTPUT: [your response]".
'''

prompt3 = '''
You are a chatbot that specializes in context comprehension, tone detection,
and empathy. Your goal is to understand both the emotional state and the 
overall context of the user’s input to ask thoughtful, open-ended questions 
that demonstrate empathy and relevance. Always analyze the user's tone 
(positive, negative, or neutral) and consider the context of their previous 
messages to form your responses. 

For example, if the user is frustrated with a specific problem they mentioned 
earlier, follow up with targeted questions related to that issue. 
If the user expresses excitement, explore the context of their excitement 
by asking about related details. 

Your role is to help the user feel understood and supported by responding 
in a way that acknowledges both their emotional tone and the specific 
situation that they are describing.

Format your output as "EMPATHY_AGENT_OUTPUT: [your response]".
For example, if the user is frustrated, your response could be:
"EMPATHY_AGENT_OUTPUT: It sounds like you're facing a challenge. 
Can you tell me more about what's been difficult?"

Always follow this format to ensure proper handling by the next agent.
'''

prompt4 = '''
You are the final agent in a chatbot pipeline. You will receive two inputs:

1. An input tagged "QUESTION_GEN_OUTPUT", 
   which is either a clarifying question or a detailed response.
2. An input tagged "EMPATHY_AGENT_OUTPUT", which is an empathy-adjusted 
   response reflecting the user's emotional state.

Your job is to merge these two inputs into a coherent final response that:
- Addresses any clarifying questions, if present, or provides the requested 
  information.
- Acknowledges the user's emotional tone and the specific context of their 
  query.
- Ensures the overall tone is empathetic, supportive, and appropriate to the 
  situation.
- Remember, you are responding to a human. Avoid long-winded responses that 
  could overwhelm the user; instead, keep your answers concise and clear.

If the first input is a clarifying question, prioritize asking the question 
while maintaining an empathetic tone.
If no clarification is needed, combine the detailed response with the context 
and tone from the empathy agent to deliver a well-rounded and sensitive reply.
'''

class App:    
    _model: ChatOpenAI
    _tools: list[BaseTool]
    _memory: MemorySaver
    _agent1_executor: CompiledGraph
    _agent2_executor: CompiledGraph
    _agent3_executor: CompiledGraph 
    _agent4_executor: CompiledGraph  

    def __init__(self):
        load_dotenv()
        self._model = ChatOpenAI(model="gpt-4")
        self._tools = [TavilySearchResults(max_results=2)]
        self._memory = MemorySaver()

        self._agent1_executor = create_react_agent(
            self._model, self._tools, state_modifier=prompt1,
            checkpointer=self._memory)
        self._agent2_executor = create_react_agent(
            self._model, self._tools, state_modifier=prompt2,
            checkpointer=self._memory)
        self._agent3_executor = create_react_agent(
            self._model, self._tools, state_modifier=prompt3,
            checkpointer=self._memory)
        self._agent4_executor = create_react_agent(
            self._model, self._tools, state_modifier=prompt4,
            checkpointer=self._memory)

    def submit_message(self, message: str, session_id: str) -> str:
        config = {"configurable": {"thread_id": session_id}}
        
        if session_id not in chat_histories:
            chat_histories[session_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "messages": []
            }

        chat_histories[session_id]["messages"].append({"role": "user", "content": message})
        
        # Agent 1: Determine if clarification is needed
        response1 = self._agent1_executor.invoke(
            {"messages": [HumanMessage(content=message)]}, config)

        clarification_result = response1["messages"][-1].content


        chat_histories[session_id]["messages"].append({"role": "system", "content": clarification_result})

        combined_message = message + clarification_result

        # Agent 2: Handle the actual response based on clarification
        response2 = self._agent2_executor.invoke(
            {"messages": [HumanMessage(content=combined_message)]},
            config
        )

        question_gen_response = response2['messages'][-1].content
        print("Question Generation Response:", question_gen_response)

        # Agent 3: Context comprehension and empathy
        response3 = self._agent3_executor.invoke(
            {"messages": [HumanMessage(content=message)]}, config)

        empathy_agent_response = response3['messages'][-1].content
        print("Empathy Agent Response:", empathy_agent_response)

        # Agent 4: Synthesize final response
        response4 = self._agent4_executor.invoke(
            {"messages": [
                HumanMessage(
                    content=question_gen_response + empathy_agent_response)
            ]
            },
            config
        )
        
        final_response = response4['messages'][-1].content
        
        chat_histories[session_id]["messages"].append({"role": "system", "content": final_response})

        return final_response

chat_app = App()



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
        chat_histories[session_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "messages": []
        }

    try:
        response = chat_app.submit_message(message, session_id)
        return jsonify({'response': response, 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/new_chat', methods=['POST'])
def new_chat():
    session_id = generate_random_session_id()
    chat_histories[session_id] = {
        "created_at": datetime.utcnow().isoformat(),
        "messages": []
    }
    return jsonify({'session_id': session_id})

@app.route('/sessions', methods=['GET'])
def get_sessions():
    """
    Returns a list of all chat sessions with their creation timestamps.
    """
    sessions = [
        {"session_id": sid, "created_at": data["created_at"]}
        for sid, data in chat_histories.items()
    ]
    return jsonify({'sessions': sessions})

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """
    Returns the chat history for a specific session.
    """
    session = chat_histories.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'session_id': session_id, 'chat_history': session['messages']})

if __name__ == '__main__':
    app.run(debug=True)
