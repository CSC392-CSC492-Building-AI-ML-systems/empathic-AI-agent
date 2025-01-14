import pandas as pd
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
from concurrent.futures import ThreadPoolExecutor

class ChatApp:
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
        self.chat_histories = {}

    def submit_message(self, message: str, session_id: str) -> str:
        config = {"configurable": {"thread_id": session_id}}

        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []

        self.chat_histories[session_id].append({"role": "user", "content": message})

        messages = [HumanMessage(content=msg["content"]) for msg in self.chat_histories[session_id]]

        response1 = self._agent1_executor.invoke(
            {"messages": messages},
            config
        )
        clarification_result = response1["messages"][-1].content

        self.chat_histories[session_id].append({"role": "system", "content": clarification_result})

        combined_message = message + clarification_result

        response2 = self._agent2_executor.invoke(
            {"messages": [HumanMessage(content=combined_message)]},
            config
        )

        final_response = response2['messages'][-1].content

        self.chat_histories[session_id].append({"role": "system", "content": final_response})

        return final_response

def process_sample(sample):
    session_id = generate_random_session_id()
    output = app.submit_message(sample, session_id)
    print(output)
    return {"input": sample, "output": output}

if __name__ == "__main__":
    df = pd.read_csv("test_dataset.csv")
    app = ChatApp()

    # we use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_sample, df["User_Input"]))

    output_df = pd.DataFrame(results)
    output_df.to_csv("output.csv", index=False)
