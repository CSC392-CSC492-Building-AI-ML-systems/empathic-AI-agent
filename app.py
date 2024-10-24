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

# Second prompt to handle the actual response
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

# Prompt for context comprehension and empathy
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

# Prompt for final synthesis agent
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
    _agent3_executor: CompiledGraph  # Context comprehension and empathy agent
    _agent4_executor: CompiledGraph  # Synthesis agent

    def __init__(self):
        self._model = ChatOpenAI(model="gpt-4")
        self._tools = [TavilySearchResults(max_results=2)]
        self._memory = MemorySaver()
        self._agent1_executer = create_react_agent(
            self._model, self._tools, state_modifier=prompt1,
            checkpointer=self._memory)
        self._agent2_executer = create_react_agent(
            self._model, self._tools, state_modifier=prompt2,
            checkpointer=self._memory)
        self._agent3_executer = create_react_agent(
            self._model, self._tools, state_modifier=prompt3,
            checkpointer=self._memory)
        self._agent4_executer = create_react_agent(
            self._model, self._tools, state_modifier=prompt4,
            checkpointer=self._memory)

    def submit_message(self, message: str, session_id: str) -> str:
        config = {"configurable": {"thread_id": session_id}}

        response1 = self._agent1_executer.invoke(
            {"messages": [HumanMessage(content=message)]}, config)

        clarification_result = response1["messages"][-1].content

        response2 = self._agent2_executer.invoke(
            {
                "messages": [
                    HumanMessage(content=message + clarification_result)
                             ]
            },
            config
        )
        question_gen_response = response2['messages'][-1].content
        print(question_gen_response)

        # Note that this agent receives the user input only,
        # without the clarification result.
        response3 = self._agent3_executer.invoke(
            {"messages": [HumanMessage(content=message)]}, config)

        empathy_agent_response = response3['messages'][-1].content
        print(empathy_agent_response)

        response4 = self._agent4_executer.invoke(
            {"messages": [
                HumanMessage(
                    content=question_gen_response + empathy_agent_response)
            ]
            },
            config
        )

        return response4['messages'][-1].content


# Sample use of the app
if __name__ == "__main__":
    app = App()
    curr_session_id = generate_random_session_id()

    print("=== WELCOME TO CHAT APP ===")
    while True:
        message = input("ME: ")
        response = app.submit_message(message, curr_session_id)
        print("SYSTEM: " + response)
