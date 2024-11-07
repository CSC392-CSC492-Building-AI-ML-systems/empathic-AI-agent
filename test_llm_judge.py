import pandas as pd
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from app import App
from utilities import generate_random_session_id
import random
import re
from collections import deque
from datetime import datetime, timedelta

load_dotenv()

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
    - 0: No follow-up
    - 1: Vague or unrelated question
    - 2: Mostly relevant
    - 3: Clear, focused follow-up

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
    - 3: Fully relevant and direct

- Engagement (0-3): How engaging is the follow-up?
    - 0: None
    - 1: Minimal engagement
    - 2: Somewhat engaging
    - 3: Highly engaging, deepens conversation

Response format:
Relevance: X
Engagement: Y
Total Score: Z""",

        3: f"""Evaluate the AI's response for detecting emotional cues and support, scoring each category from 0-4.

User input: "{user_input}"
AI's response: "{actual_response}"

- Empathy (0-4): How well does the AI detect and respond to emotions?
    - 0: No acknowledgment
    - 1: Minimal acknowledgment
    - 2: Some recognition
    - 3: Strong recognition
    - 4: Fully acknowledges and responds empathetically

- Supportiveness (0-4): How supportive is the AI?
    - 0: No support
    - 1: Minimal support
    - 2: Some support
    - 3: Supportive, uses empathy
    - 4: Very supportive, offers help

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
    - 2: Acknowledges and partially corrects
    - 3: Fully acknowledges and corrects politely

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
            actual_response = app.submit_message(row['User_Input'], session_id)
            
            time.sleep(random.uniform(1.5, 2.5))
            
            evaluation = evaluate_with_openai(
                row['User_Story'], 
                row['User_Input'], 
                actual_response, 
                chat,
                rate_limiter
            )
            
            return {
                "ID": row["ID"],
                "Batch_ID": batch_id,
                "User_Story": row['User_Story'],
                "User_Input": row['User_Input'],
                "Expected_Response": row['Expected_Response'],
                "Actual_Response": actual_response,
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
                    "Actual_Response": None,
                    "Evaluation": f"Error after {max_retries} attempts: {str(e)}",
                    "Emotion": row.get('Emotion', None),
                    "Timestamp": datetime.now().isoformat(),
                    "Attempts": max_retries
                }

def save_results(results, input_csv_path, batch_id):
    """
    Saves test results to a CSV file with batch tracking.
    """
    results_df = pd.DataFrame(results)
    input_filename = os.path.basename(input_csv_path)
    output_csv_path = f"results_llm_judge_{batch_id}_{input_filename}"
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

def run_tests_and_save_results(input_csv_path, app, batch_size=3):
    """
    Runs test cases in smaller batches with improved rate limiting.
    """
    print(f"Loading test cases from {input_csv_path}...")
    test_cases = pd.read_csv(input_csv_path)
    print(f"Loaded {len(test_cases)} test cases.")
    
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
        print(f"Processing batch {i//batch_size + 1} of {(len(test_cases) + batch_size - 1)//batch_size}")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for _, row in batch.iterrows():
                session_id = generate_random_session_id()
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
    app = App()
    input_csv_path = "test_dataset.csv"
    run_tests_and_save_results(input_csv_path, app)