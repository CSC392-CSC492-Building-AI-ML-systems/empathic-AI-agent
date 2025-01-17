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
import sqlite3
import asyncio
import aiohttp


def init_db():
    conn = sqlite3.connect("database_temp.db")
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

app = Flask(__name__)
load_dotenv()  

LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

prompt1 = '''
Analyze the user's request and determine if it requires clarification 
due to ambiguity.
When it comes to examples of ambiguous questions, they can typically include a 
number of characteristics such as:

- Vagueness
- Not clearly defining the subject
- More than one meaning
- Asking for several responses
- Phrase or word (like "links") that implies a certain subject matter but could 
  still be ambiguous (e.g., servers as computers or in food service)

If clarification is needed:
1. Respond with "CLARIFICATION_NEEDED: " followed by the clarifying question.
2. Example: "CLARIFICATION_NEEDED: Would you like to know more about servers in the context 
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
overall context of the user's input to ask thoughtful, open-ended questions 
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

1. An input tagged "QUESTION_AGENT_OUTPUT", 
   which is either a clarifying question or a detailed response.
2. An input tagged "EMPATHY_AGENT_OUTPUT", which is an empathy-adjusted 
   response reflecting the user's emotional state.

Your job is to merge these two inputs into a coherent final response that:
- Addresses any clarifying questions, if present, or provides the requested 
  information. Note that you should always prioritize asking questions if
  a major ambiguity has been detected, and should not provide specific
  information if the question is being asked.
- Acknowledges the user's emotional tone and the specific context of their 
  query.
- Ensures the overall tone is empathetic, supportive, and appropriate to the 
  situation.
- Remember, you are responding to a human. Avoid long-winded responses that 
  could overwhelm the user; instead, keep your answers concise and clear.

If the first input is a clarifying question, prioritize asking the question 
while maintaining an empathetic tone. Do not respond further if asking a 
question!
If no clarification is needed, combine the detailed response with the context 
and tone from the empathy agent to deliver a well-rounded and sensitive reply.
Ensure that no tags are present in your output.
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
        
        conn = sqlite3.connect("database_temp.db")
        cur = conn.cursor()
        
        # Store user message
        cur.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "user", message))
        conn.commit()

        # Agent 1: Determine if clarification is needed
        response1 = self._agent1_executor.invoke(
            {"messages": [HumanMessage(content=message)]}, config)
        clarification_result = response1["messages"][-1].content
        
        # Store Agent 1 output
        cur.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "agent1", clarification_result))
        conn.commit()

        combined_message = message + clarification_result

        # Agent 2: Handle the actual response based on clarification
        response2 = self._agent2_executor.invoke(
            {"messages": [HumanMessage(content=combined_message)]},
            config
        )
        question_gen_response = response2['messages'][-1].content
        
        # Store Agent 2 output
        cur.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "agent2", question_gen_response))
        conn.commit()

        # Agent 3: Context comprehension and empathy
        response3 = self._agent3_executor.invoke(
            {"messages": [HumanMessage(content=message)]}, config)
        empathy_agent_response = response3['messages'][-1].content
        
        # Store Agent 3 output
        cur.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "agent3", empathy_agent_response))
        conn.commit()

        # Agent 4: Synthesize final response
        response4 = self._agent4_executor.invoke(
            {"messages": [
                HumanMessage(
                    content=question_gen_response + empathy_agent_response)
            ]},
            config
        )
        final_response = response4['messages'][-1].content
        
        # Store final response
        cur.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "agent4", final_response))
        conn.commit()
        
        conn.close()

        # Comment out to hide intermediate responses (needed for demo purposes)
        demo_response = "**CLARIFICATION GENERATION AGENT:**" + "\n" + clarification_result + "\n\n"
        demo_response += "**QUESTION ASKING AGENT:**" + "\n" + question_gen_response + "\n\n"
        demo_response += "**CONTEXT COMPREHENSION AND EMPATHY AGENT:**" + "\n" + empathy_agent_response + "\n\n"
        demo_response += "**FINAL SYNTHESIS AGENT:**" + "\n" + final_response
        return demo_response
    
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
        conn = sqlite3.connect("database_temp.db")
        cur = conn.cursor()
        cur.execute("INSERT INTO chat_sessions (session_id, created_at) VALUES (?, ?)",
                   (session_id, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

    try:
        response = chat_app.submit_message(message, session_id)
        return jsonify({'response': response, 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/new_chat', methods=['POST'])
def new_chat():
    session_id = generate_random_session_id()
    conn = sqlite3.connect("database_temp.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO chat_sessions (session_id, created_at) VALUES (?, ?)",
               (session_id, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return jsonify({'session_id': session_id})

@app.route('/sessions', methods=['GET'])
def get_sessions():
    conn = sqlite3.connect("database_temp.db")
    cur = conn.cursor()
    cur.execute("SELECT session_id, created_at FROM chat_sessions")
    sessions = [{"session_id": row[0], "created_at": row[1]} for row in cur.fetchall()]
    conn.close()
    return jsonify({'sessions': sessions})

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    conn = sqlite3.connect("database_temp.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT role, content 
        FROM messages 
        WHERE session_id = ? AND role IN ('user', 'agent4')
        ORDER BY id ASC
    """, (session_id,))
    messages = [{"role": row[0], "content": row[1]} for row in cur.fetchall()]
    conn.close()
    
    if not messages:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'session_id': session_id, 'chat_history': messages})

if __name__ == '__main__':
    app.run(debug=True)