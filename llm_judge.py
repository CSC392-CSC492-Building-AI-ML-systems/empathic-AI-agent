import argparse
import pandas as pd
import os
import time
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from concurrent.futures import ThreadPoolExecutor, as_completed
from utilities import generate_random_session_id
import random
import re
from collections import deque
from datetime import datetime, timedelta

load_dotenv()

LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

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

        # Agent 1: Determine if clarification is needed
        response1 = self._agent1_executor.invoke(
            {"messages": [HumanMessage(content=message)]}, config)
        clarification_result = response1["messages"][-1].content

        combined_message = message + clarification_result

        # Agent 2: Handle the actual response based on clarification
        response2 = self._agent2_executor.invoke(
            {"messages": [HumanMessage(content=combined_message)]},
            config
        )
        question_gen_response = response2['messages'][-1].content

        # Agent 3: Context comprehension and empathy
        response3 = self._agent3_executor.invoke(
            {"messages": [HumanMessage(content=message)]}, config)
        empathy_agent_response = response3['messages'][-1].content

        # Agent 4: Synthesize final response
        response4 = self._agent4_executor.invoke(
            {"messages": [
                HumanMessage(
                    content=question_gen_response + empathy_agent_response)
            ]},
            config
        )
        final_response = response4['messages'][-1].content
        
        # Capture outputs of each agent step-by-step
        agent1_output = clarification_result
        agent2_output = response2["messages"][-1].content
        agent3_output = response3["messages"][-1].content
        agent4_output = response4["messages"][-1].content
    
        # Return the final combined response along with each individual agent output
        return agent1_output, agent2_output, agent3_output, agent4_output

class RateLimiter:
    def __init__(self, requests_per_minute=3500, requests_per_day=200000, token_limit=9000):  # Reduced from 10000
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.token_limit = token_limit
        self.minute_requests = deque()
        self.day_requests = deque()
        self.token_usage = deque()
        self.last_reset = datetime.now()
    
    def wait_if_needed(self, estimated_tokens=0):
        current_time = datetime.now()
        
        if current_time - self.last_reset >= timedelta(minutes=1):
            self.token_usage.clear()
            self.last_reset = current_time
            time.sleep(1) 
        
        while self.minute_requests and self.minute_requests[0] < current_time - timedelta(minutes=1):
            self.minute_requests.popleft()
        while self.day_requests and self.day_requests[0] < current_time - timedelta(days=1):
            self.day_requests.popleft()
        
        current_tokens = sum(self.token_usage)
        estimated_tokens = int(estimated_tokens * 1.2) 
        

        if current_tokens + estimated_tokens > self.token_limit:
            sleep_time = 61 - (current_time - self.last_reset).total_seconds() 
            if sleep_time > 0:
                time.sleep(sleep_time + random.uniform(0.5, 1.0))  
                self.token_usage.clear()
                self.last_reset = datetime.now()
        
        if len(self.minute_requests) >= self.requests_per_minute:
            sleep_time = (self.minute_requests[0] + timedelta(minutes=1) - current_time).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time + random.uniform(0.5, 1.0))
        
        if len(self.day_requests) >= self.requests_per_day:
            sleep_time = (self.day_requests[0] + timedelta(days=1) - current_time).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time + random.uniform(0.5, 1.0))

        current_time = datetime.now()
        self.minute_requests.append(current_time)
        self.day_requests.append(current_time)
        self.token_usage.append(estimated_tokens)

def get_evaluation_prompt(user_story, user_input, actual_response):
    """
    Returns the appropriate evaluation prompt based on the user story.
    """
    prompts = {
        1: f"""Evaluate the AI's response for detecting ambiguity and asking clarifying questions. Score each category from 0-3 based on the criteria below.

User input: "{user_input}"
AI's response: "{actual_response}"

- Clarity (0-3): How clearly does the AI address the user input?
    - 0: Off-topic or unclear
    - 1: Some relevance, lacks clarity
    - 2: Mostly clear, minor gaps
    - 3: Very clear, directly addresses input

- Follow-Up (0-3): How well does the AI ask relevant follow-up questions?
    - 0: No follow-up, fails to address any ambiguity
    - 1: Vague or unrelated question
    - 2: Mostly relevant follow-up, but lacks precision
    - 3: Clear, relevant follow-up question that addresses potential ambiguity

Response format:
Clarity: X
Follow-Up: Y
Total Score: Z""",

        2: f"""Evaluate for contextually relevant follow-up questions, scoring each category from 0-3.

User input: "{user_input}"
AI's response: "{actual_response}"

- Relevance (0-3): How well does the AI respond to user input?
    - 0: Off-topic
    - 1: Partially relevant
    - 2: Mostly relevant, minor gaps
    - 3: Fully relevant, direct and concise

- Engagement (0-3): How engaging is the follow-up question?
    - 0: No follow-up or engagement, does not encourage user to continue
    - 1: Minimal engagement
    - 2: Somewhat engaging
    - 3: Highly engaging, deepens the conversation with follow-up

Response format:
Relevance: X
Engagement: Y
Total Score: Z""",

        3: f"""Evaluate the AI's response for detecting emotional cues and support, scoring each category from 0-3.

User input: "{user_input}"
AI's response: "{actual_response}"

- Empathy (0-3): How well does the AI detect and respond to emotions?
    - 0: No acknowledgment
    - 1: Minimal acknowledgment
    - 2: Some recognition
    - 3: Fully acknowledges and responds empathetically

- Supportiveness (0-3): How supportive is the AI?
    - 0: No support
    - 1: Minimal support
    - 2: Some support but lacks clear follow-up
    - 3: Very supportive, offers clear follow-up

Response format:
Empathy: X
Supportiveness: Y
Total Score: Z""",

        4: f"""Evaluate the AI's handling of misunderstandings, scoring each category from 0-3.

User input: "{user_input}"
AI's response: "{actual_response}"

- Politeness (0-3): How polite is the AI?
    - 0: Rude
    - 1: Polite but forced
    - 2: Mostly polite
    - 3: Very polite

- Correction Handling (0-3): How well does the AI handle misunderstandings?
    - 0: Ignores or repeats error
    - 1: Acknowledges but lacks clarity
    - 2: Partially corrects but needs clearer follow-up
    - 3: Fully acknowledges and politely corrects by asking a follow-up

Response format:
Politeness: X
Correction Handling: Y
Total Score: Z"""
    }
    
    try:
        user_story = int(user_story)
    except ValueError:
        return "Invalid user story format"
    
    return prompts.get(user_story, "User story not found")

def estimate_tokens(text):
    # More conservative token estimation: ~3 characters per token
    return len(text) // 3

def evaluate_with_openai(user_story, user_input, actual_response, chat, rate_limiter, max_retries=5, base_delay=2):
    """
    Uses OpenAI to evaluate the AI's response with improved rate limiting and exponential backoff.
    """
    prompt = get_evaluation_prompt(user_story, user_input, actual_response)
    estimated_tokens = estimate_tokens(prompt)
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed(estimated_tokens)
            response = chat.invoke(prompt)
            return response.content.strip()
        
        except Exception as e:
            error_message = str(e).lower()
            
            if re.search(r"rate limit|429|rate_limit_exceeded|tokens per min", error_message):
                # Extract wait time and add buffer
                wait_match = re.search(r"try again in (\d+\.?\d*)s", error_message)
                if wait_match:
                    wait_time = float(wait_match.group(1)) + 1  # Add 1 second buffer
                else:
                    wait_time = min((2 ** attempt) * base_delay + random.uniform(1.0, 2.0), 60)
                
                print(f"Rate limit hit. Waiting {wait_time:.2f}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                continue
            
            print(f"Non-rate-limit error: {e}")
            return f"Error: {str(e)}"
    
    return "Evaluation failed after maximum retries"

def process_test_case(row, app, chat, rate_limiter, session_id, batch_id):
    """
    Processes a single test case with improved rate limiting.
    """
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            # Process each agent's output individually
            agent1_output, agent2_output, agent3_output, agent4_output = app.submit_message(row['User_Input'], session_id)
            
            time.sleep(random.uniform(1.5, 2.5))
            
            evaluation = evaluate_with_openai(
                row['User_Story'], 
                row['User_Input'], 
                agent4_output, 
                chat,
                rate_limiter
            )
            
            return {
                "ID": row["ID"],
                "Batch_ID": batch_id,
                "User_Story": row['User_Story'],
                "User_Input": row['User_Input'],
                "Expected_Response": row['Expected_Response'],
                "Agent1_Output": agent1_output,
                "Agent2_Output": agent2_output,
                "Agent3_Output": agent3_output,
                "Agent4_Output": agent4_output,
                "Evaluation": evaluation,
                "Emotion": row.get('Emotion', None),
                "Timestamp": datetime.now().isoformat(),
                "Attempts": attempt + 1
            }
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * base_delay + random.uniform(1.0, 2.0)
                print(f"Error processing test case {row['ID']}, retrying in {wait_time:.2f}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"Failed to process test case {row['ID']} after {max_retries} attempts: {e}")
                return {
                    "ID": row["ID"],
                    "Batch_ID": batch_id,
                    "User_Story": row['User_Story'],
                    "User_Input": row['User_Input'],
                    "Expected_Response": row['Expected_Response'],
                    "Agent1_Output": None,
                    "Agent2_Output": None,
                    "Agent3_Output": None,
                    "Agent4_Output": None,
                    "Evaluation": f"Error after {max_retries} attempts: {str(e)}",
                    "Emotion": row.get('Emotion', None),
                    "Timestamp": datetime.now().isoformat(),
                    "Attempts": max_retries
                }

def parse_evaluation_column(evaluation_text):
    # Define the pattern to match scores like "Clarity: 3"
    pattern = r"(Clarity|Follow-Up|Relevance|Engagement|Empathy|Supportiveness|Politeness|Correction Handling|Total Score): (\d+)"
    scores = {}
    
    for match in re.findall(pattern, evaluation_text):
        score_name, score_value = match
        scores[score_name] = int(score_value)
    
    return scores

def save_results(results, input_csv_path, batch_id):
    results_df = pd.DataFrame(results)
    input_filename = os.path.basename(input_csv_path)
    output_csv_path = f"results_llm_judge_{batch_id}_{input_filename}"
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    # Initialize dictionary to store parsed scores for each category
    all_scores = {
        "Clarity": [],
        "Follow-Up": [],
        "Relevance": [],
        "Engagement": [],
        "Empathy": [],
        "Supportiveness": [],
        "Politeness": [],
        "Correction Handling": [],
        "Total Score": []
    }

    # Parse "Evaluation" column for each score and add to respective lists in all_scores
    for evaluation_text in results_df["Evaluation"]:
        evaluation_scores = parse_evaluation_column(evaluation_text)
        for category in all_scores.keys():
            if category in evaluation_scores:
                all_scores[category].append(evaluation_scores[category])

    # Create summary data by calculating sum and average for each category
    summary_data = {
        "Category": [],
        "Sum": [],
        "Average": []
    }

    for category, scores in all_scores.items():
        summary_data["Category"].append(category)
        summary_data["Sum"].append(sum(scores))
        summary_data["Average"].append(sum(scores) / len(scores) if scores else 0)

    # Convert summary_data to DataFrame and save as CSV with row-per-category format
    summary_df = pd.DataFrame(summary_data)
    summary_output_path = f"summary_llm_judge_{batch_id}_{input_filename}"
    summary_df.to_csv(summary_output_path, index=False)
    print(f"Summary saved to {summary_output_path}")

def run_tests_and_save_results(input_csv_path, app, batch_size=3, row_number=None, retries=1):
    """
    Runs test cases in smaller batches with improved rate limiting.
    If row_number is provided, only that row is processed.
    If retries are provided, the specified number of retries is used.
    """
    print(f"Loading test cases from {input_csv_path}...")
    test_cases = pd.read_csv(input_csv_path)
    print(f"Loaded {len(test_cases)} test cases.")
    
    if row_number is not None:
        row_number -= 1  # Adjust for 1-based indexing
        if not (0 <= row_number < len(test_cases)):
            print("Error: Row number is out of range.")
            return
        test_cases = test_cases.iloc[[row_number]]
        print(f"Processing only row {row_number + 1}.")  # Display the original row number

    results = []
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    chat = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4",
        temperature=0
    )
    rate_limiter = RateLimiter()
    
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases.iloc[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {(len(test_cases) + batch_size - 1) // batch_size}")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for _, row in batch.iterrows():
                session_id = generate_random_session_id()
                for _ in range(retries):
                    futures.append(
                        executor.submit(
                            process_test_case, 
                            row, 
                            app, 
                            chat,
                            rate_limiter,
                            session_id, 
                            batch_id
                        )
                    )
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if len(results) % batch_size == 0:
                        save_results(results, input_csv_path, batch_id)
                except Exception as e:
                    print(f"Error processing future: {e}")
        
        time.sleep(random.uniform(2.5, 3.5))
    
    # Save final results
    save_results(results, input_csv_path, batch_id)
    print(f"All results saved. Batch ID: {batch_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM judge tests on a CSV dataset.")
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument("row", nargs="?", type=int, help="Row number from the input CSV file to process (optional).")
    parser.add_argument("retries", nargs="?", type=int, default=1, help="Number of retries for the specified row (default: 1).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File '{args.csv_path}' not found.")
    else:
        app = App()
        run_tests_and_save_results(args.csv_path, app, row_number=args.row, retries=args.retries)
