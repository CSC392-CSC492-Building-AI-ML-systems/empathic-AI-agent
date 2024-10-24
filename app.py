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

load_dotenv()  # Load environment variables from .env file
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# First prompt to determine if clarification is needed
prompt1 = '''
Analyze the user's request and determine if it requires clarification due to ambiguity.
If clarification is needed:
1. Respond with "CLARIFICATION_NEEDED: " followed by the clarifying question.
2. Example: "CLARIFICATION_NEEDED: Are you asking about servers in the context of computers or food service?"

If no clarification is needed:
1. Respond with "NO_CLARIFICATION_NEEDED"

Always provide only one of these two responses, with no additional text.
'''

# Second prompt to handle the actual response
prompt2 = '''
You are an assistant that provides information based on the user's request.
You will receive input in one of two formats:

1. A user query followed by "NO_CLARIFICATION_NEEDED"
   In this case, provide a detailed response to the user's query.

2. A user query followed by "CLARIFICATION_NEEDED: [question]"
   In this case, ask the clarifying question provided.

Maintain a conversational tone and ensure your response is appropriate to the input received.
'''


class App:
    _model: ChatOpenAI
    _tools: list[BaseTool]
    _memory: MemorySaver
    _agent1_executor: CompiledGraph
    _agent2_executor: CompiledGraph

    def __init__(self):
        self._model = ChatOpenAI(model="gpt-4")
        self._tools = [TavilySearchResults(max_results=2)]
        self._memory = MemorySaver()
        self._agent1_executer = create_react_agent(self._model, self._tools, state_modifier=prompt1, checkpointer=self._memory)
        self._agent2_executer = create_react_agent(self._model, self._tools, state_modifier=prompt2, checkpointer=self._memory)

    def submit_message(self, message: str, session_id: str) -> str:
        config = {"configurable": {"thread_id": session_id}}

        response1 = self._agent1_executer.invoke({"messages": [HumanMessage(content=message)]}, config)

        clarification_result = response1["messages"][-1].content

        response2 = self._agent2_executer.invoke({"messages": [HumanMessage(content=message + clarification_result)]}, config)

        return response2['messages'][-1].content


# Sample use of the app
if __name__ == "__main__":
    app = App()
    curr_session_id = generate_random_session_id()

    print("=== WELCOME TO CHAT APP ===")
    while True:
        message = input("ME: ")
        response = app.submit_message(message, curr_session_id)
        print("SYSTEM: " + response)
