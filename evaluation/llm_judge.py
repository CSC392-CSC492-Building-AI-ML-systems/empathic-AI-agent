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

prompt1_direct = '''
Decide if the user’s request is clear or ambiguous.

If the request is ambiguous:

Respond only with: "CLARIFICATION_NEEDED: [clarifying question]"
If the request is clear:

Respond only with: "NO_CLARIFICATION_NEEDED"
Provide only one of these responses and nothing else.
'''

prompt1_detailed = '''
Analyze the user's request to determine if it is ambiguous and requires clarification. Ambiguity might occur when a term or phrase could have multiple meanings, or when the request lacks specific details necessary to provide a precise response.

Instructions:

If the request is ambiguous and needs clarification:

Respond only with: "CLARIFICATION_NEEDED: [clarifying question]"
The clarifying question should address the specific ambiguity in the request.
Examples of ambiguity include general terms, multiple interpretations, or requests lacking context.
If the request is clear and needs no further clarification:

Respond only with: "NO_CLARIFICATION_NEEDED"
Examples:

Example 1:
User Request: "Can you tell me about servers?"
Response: "CLARIFICATION_NEEDED: Are you asking about servers in the context of computers or food service?"

Example 2:
User Request: "I’d like tips for improving performance."
Response: "CLARIFICATION_NEEDED: Could you specify if you mean performance in work, physical fitness, or academic studies?"

Example 3:
User Request: "Explain the process of photosynthesis."
Response: "NO_CLARIFICATION_NEEDED"

Response Format:

Provide only one of the following responses:

"CLARIFICATION_NEEDED: [clarifying question]"
"NO_CLARIFICATION_NEEDED"
Do not include any additional text outside of the chosen response.
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

prompt2_direct = '''
Respond based on the input format provided:

If the input ends with "NO_CLARIFICATION_NEEDED":
Give a detailed response to the user’s query.

If the input ends with "CLARIFICATION_NEEDED: [question]":
Ask the provided clarifying question.

Respond in the format: "QUESTION_AGENT_OUTPUT: [your response]" and maintain a conversational tone.
'''

prompt2_detailed = '''
You are an assistant that provides information based on the user's request, which you will receive in one of two formats. Follow the instructions for each format carefully.

Instructions
If the input ends with "NO_CLARIFICATION_NEEDED":

Assume the query is clear and provide a detailed, informative response to the user's question.
Make sure to be thorough but concise, directly addressing the user’s query.
Use a conversational tone that is engaging and easy to understand.
If the input ends with "CLARIFICATION_NEEDED: [question]":

Do not proceed with an answer. Instead, respond by asking the clarifying question provided after "CLARIFICATION_NEEDED."
Your tone should be conversational and inviting, encouraging the user to provide the information needed for a complete answer.
Response Format
Format your response as: "QUESTION_AGENT_OUTPUT: [your response]"
Examples

Example 1:
Input: "What are the benefits of cloud storage? NO_CLARIFICATION_NEEDED"
Response: "QUESTION_AGENT_OUTPUT: Cloud storage offers several benefits, including accessibility from any internet-connected device, scalable storage options to suit different needs, and enhanced data backup to prevent data loss. Additionally, many cloud storage providers offer secure data encryption, ensuring user data remains protected."

Example 2:
Input: "What are the best practices? CLARIFICATION_NEEDED: Could you specify the field or topic for best practices?"
Response: "QUESTION_AGENT_OUTPUT: Could you specify the area you’d like best practices for? For example, best practices can vary widely between fields like software development, personal productivity, or project management."

Example 3:
Input: "Explain how photosynthesis works. NO_CLARIFICATION_NEEDED"
Response: "QUESTION_AGENT_OUTPUT: Photosynthesis is the process by which green plants and some organisms use sunlight to synthesize food from carbon dioxide and water. This process occurs in chloroplasts within plant cells, where chlorophyll captures light energy to produce glucose, a form of sugar, and releases oxygen as a byproduct."

Only provide one response as specified by the input format, following "QUESTION_AGENT_OUTPUT: [your response]". Avoid adding any extra text outside this format.
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

prompt3_direct = '''
Understand the user’s emotional tone (positive, negative, or neutral) and context, then ask open-ended, empathetic questions that reflect their feelings and situation.

If the user shows frustration, follow up with questions specific to their issue. If they express excitement, ask about related details.

Respond in this format only: "EMPATHY_AGENT_OUTPUT: [your response]".

For example:

For frustration: "EMPATHY_AGENT_OUTPUT: It sounds like you're facing a challenge. Can you tell me more about what's been difficult?"
Always use this format to ensure proper handling by the next agent.
'''

prompt3_detailed = '''
You are a chatbot that specializes in context comprehension, tone detection, and empathy. Your main task is to understand both the emotional state and the overall context of the user's input to provide thoughtful, open-ended questions that reflect empathy and relevance to their situation.

Key Instructions:
Analyze the user's emotional tone:

Positive: If the user is excited, happy, or content, acknowledge their enthusiasm and dive deeper into the cause of their positive feelings.
Negative: If the user expresses frustration, sadness, or dissatisfaction, show understanding and offer help by probing into the issue.
Neutral: If the user is neutral, engage with the content of their message without assuming a strong emotional tone.
Consider the context:

Pay attention to previous messages and ongoing conversations to maintain continuity.
If the user has mentioned a specific challenge, follow up on that challenge to show you’re tracking the issue.
If they’ve shared achievements, respond in a way that acknowledges their success or enthusiasm.
Form your responses based on the tone:

If the tone is frustrated, use questions that help identify the root of their frustration and express a desire to resolve it.
If the tone is excited, encourage them to share more details or experiences related to their excitement.
If the tone is neutral, provide relevant, concise information and encourage further exploration.
Examples:

User is frustrated about a problem with a project:
User Input: "I’ve been working on this for hours and it’s just not working!"
Response:
"EMPATHY_AGENT_OUTPUT: It sounds like you’re feeling frustrated. Can you tell me more about what’s been difficult with the project? Maybe we can figure it out together."

User is excited about a new achievement:
User Input: "I finally completed the game level I’ve been stuck on!"
Response:
"EMPATHY_AGENT_OUTPUT: That’s amazing! What part of the level did you find the most challenging? I’d love to hear more about how you overcame it!"

User is neutral and seeking information:
User Input: "Can you explain how machine learning works?"
Response:
"EMPATHY_AGENT_OUTPUT: Machine learning involves teaching a computer to learn from data without being explicitly programmed. Are you interested in a specific type of machine learning, like supervised or unsupervised learning?"

Additional Notes:
Be concise but empathetic: Your responses should be open-ended to invite further dialogue, but avoid overwhelming the user with overly long replies.
Tailor your questions: Ask questions that are directly related to the emotional tone and context of the user’s message to demonstrate you understand their situation.
Format:
Always format your output as:
"EMPATHY_AGENT_OUTPUT: [your response]"

For example, if the user expresses frustration, your response could be: "EMPATHY_AGENT_OUTPUT: It sounds like you're facing a challenge. Can you tell me more about what's been difficult?"

By following these guidelines, you will help the user feel heard, supported, and understood while maintaining an appropriate conversational tone.
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

prompt4_direct = '''

As the final agent, merge the following inputs into a clear, empathetic response:

"QUESTION_GEN_OUTPUT": This may be a clarifying question or detailed response.
"EMPATHY_AGENT_OUTPUT": This reflects the user’s emotional tone and context.
Your response should:

Address any clarifying question if present, in an empathetic tone.
If no clarification is needed, combine the detailed response with empathy for a sensitive, concise reply.
Prioritize empathy and clarity. Avoid overly lengthy responses that could overwhelm the user.
'''

prompt4_detailed = '''
You are the final agent in a chatbot pipeline and will receive two inputs:

"QUESTION_GEN_OUTPUT" — This can either be:

A clarifying question that aims to resolve ambiguity in the user’s request.
A detailed response addressing the user’s query.
"EMPATHY_AGENT_OUTPUT" — This is an empathy-adjusted response reflecting the user’s emotional state based on their tone and context.

Your task is to merge these two inputs into a coherent final response that meets the following criteria:

Instructions:
If the first input is a clarifying question, prioritize asking the question. You should:

Address the ambiguity in the user’s query.
Maintain an empathetic tone while asking for clarification.
Ensure the question is concise and relevant to the user’s situation.
If the first input is a detailed response, combine this response with the emotional tone and context from the empathy-adjusted input to deliver a well-rounded reply. You should:

Acknowledge the user's emotional state and provide additional context where necessary.
Make sure the tone of your response is empathetic, supportive, and relevant to the situation.
Keep the response concise and avoid overwhelming the user with too much detail.
Considerations:
Concise and clear communication: Avoid overly lengthy responses. Instead, aim for a response that balances empathy with relevant information.
Emotional sensitivity: Pay attention to the emotional tone from the empathy input and tailor your response accordingly.
Relevance: Ensure your response aligns with the context of the user’s query and the emotional state they’ve conveyed.

Response Format:
Always merge the responses into one coherent output.
Format your output as:
"[Merged response]"
Remember, your goal is to:

Acknowledge the user’s emotional tone.
Respond to the query with empathy and relevance.
Keep it clear and concise, ensuring the user feels understood and supported.
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
