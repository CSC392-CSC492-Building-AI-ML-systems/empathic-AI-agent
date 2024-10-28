import pandas as pd
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from app import App  # Import AI application class from app.py
from utilities import generate_random_session_id
import random
import re

load_dotenv()

chat = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",
    temperature=0  # You can adjust temperature for variability in responses
)

def evaluate_with_openai(user_story, user_input, actual_response, max_retries=5):
    """
    Uses OpenAI to evaluate the AI's actual response based on the user story acceptance criteria.
    """
    # Define the prompts for each user story
    prompts = {
        1: f"""Evaluate the AI's response for detecting ambiguity and asking clarifying questions. Score each category from 0-3 based on the criteria below.

User input: "{user_input}"
AI's response: "{actual_response}"

- Clarity (0-3): How clearly does the AI address the user input?
    - 0: Off-topic or unclear.
    - 1: Some relevance, lacks clarity.
    - 2: Mostly clear, minor gaps.
    - 3: Very clear, directly addresses input.

- Follow-Up (0-3): How well does the AI ask relevant follow-up questions?
    - 0: No follow-up.
    - 1: Vague or unrelated question.
    - 2: Mostly relevant.
    - 3: Clear, focused follow-up.

Response format:
Clarity: X
Follow-Up: Y
Total Score: Z""",

        2: f"""Evaluate for contextually relevant follow-up questions, scoring each category from 0-3.

User input: "{user_input}"
AI's response: "{actual_response}"

- Relevance (0-3): How well does the AI respond to user input?
    - 0: Off-topic.
    - 1: Partially relevant.
    - 2: Mostly relevant, minor gaps.
    - 3: Fully relevant and direct.

- Engagement (0-3): How engaging is the follow-up?
    - 0: None.
    - 1: Minimal engagement.
    - 2: Somewhat engaging.
    - 3: Highly engaging, deepens conversation.

Response format:
Relevance: X
Engagement: Y
Total Score: Z""",

        3: f"""Evaluate the AI’s response for detecting emotional cues and support, scoring each category from 0-4.

User input: "{user_input}"
AI's response: "{actual_response}"

- Empathy (0-4): How well does the AI detect and respond to emotions?
    - 0: No acknowledgment.
    - 1: Minimal acknowledgment.
    - 2: Some recognition.
    - 3: Strong recognition.
    - 4: Fully acknowledges and responds empathetically.

- Supportiveness (0-4): How supportive is the AI?
    - 0: No support.
    - 1: Minimal support.
    - 2: Some support.
    - 3: Supportive, uses empathy.
    - 4: Very supportive, offers help.

Response format:
Empathy: X
Supportiveness: Y
Total Score: Z""",

        4: f"""Evaluate the AI’s handling of misunderstandings, scoring each category from 0-3.

User input: "{user_input}"
AI's response: "{actual_response}"

- Politeness (0-3): How polite is the AI?
    - 0: Rude.
    - 1: Polite but forced.
    - 2: Mostly polite.
    - 3: Very polite.

- Correction Handling (0-3): How well does the AI handle misunderstandings?
    - 0: Ignores or repeats error.
    - 1: Acknowledges but lacks clarity.
    - 2: Acknowledges and partially corrects.
    - 3: Fully acknowledges and corrects politely.

Response format:
Politeness: X
Correction Handling: Y
Total Score: Z"""
    }

    # Convert the user_story to an integer
    try:
        user_story = int(user_story)
    except ValueError:
        return "User story not found"

    # Get the corresponding prompt for the user story
    prompt = prompts.get(user_story, "")

    # If no prompt is available for the user story, return "User story not found"
    if not prompt:
        return "User story not found"

    attempt = 0
    while attempt < max_retries:
        try:
            # Use ChatOpenAI's invoke method to generate a response
            response = chat.invoke(prompt)
            return response.content.strip()
        
        except Exception as e:
            error_message = str(e).lower()
            
            # Check for rate limit or similar errors
            if re.search(r"rate limit|429|rate_limit_exceeded|tokens per min", error_message):
                # Exponential backoff with randomized delay
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"Rate limit error in evaluation. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1})")
                time.sleep(wait_time)
                attempt += 1
            else:
                # Log and return a non-rate-limit error message
                print(f"Non-rate-limit error in evaluation: {e}")
                return f"Error: Non-rate-limit error: {str(e)}"

    # If max retries are exhausted due to rate limits, return a standardized message
    print(f"Failed to complete evaluation after {max_retries} attempts due to rate limit.")
    return "Evaluation failed due to repeated rate limits"

def process_test_case(row, app, session_id, max_retries=5):
    """
    Processes a single test case by sending the input to the AI and evaluating the response.
    Retries if a rate limit error occurs, with extended backoff and random delay.
    """
    user_story = row['User_Story']
    user_input = row['User_Input']
    expected_response = row['Expected_Response']
    emotion = row.get('Emotion', None)

    attempt = 0

    while attempt < max_retries:
        try:
            # Get actual response from AI
            actual_response = app.submit_message(user_input, session_id)
            
            # Use OpenAI to evaluate the response based on the user story
            evaluation = evaluate_with_openai(user_story, user_input, actual_response)
            
            # Prepare the result to be written to the CSV
            return {
                "ID": row["ID"],
                "User_Story": user_story,
                "User_Input": user_input,
                "Expected_Response": expected_response,
                "Actual_Response": actual_response,
                "Evaluation": evaluation,
                "Emotion": emotion
            }
        
        except Exception as e:
            error_message = str(e).lower()
            
            # Check for rate limit error using a more comprehensive pattern match
            if re.search(r"rate limit|429|rate_limit_exceeded|tokens per min", error_message):
                # Exponential backoff with a random factor to avoid synchronized retries
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)  # Randomized delay
                print(f"Rate limit error. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1})")
                time.sleep(wait_time)
                attempt += 1
            else:
                # Log and return a standard error message for non-rate-limit errors
                print(f"Non-rate-limit error processing test case: {e}")
                return {
                    "ID": row["ID"],
                    "User_Story": user_story,
                    "User_Input": user_input,
                    "Expected_Response": expected_response,
                    "Actual_Response": None,
                    "Evaluation": f"Error: Non-rate-limit error: {str(e)}",
                    "Emotion": emotion
                }

    # If max retries reached due to rate limits, log and return a standardized message
    print(f"Failed to process test case after {max_retries} attempts due to rate limit.")
    return {
        "ID": row["ID"],
        "User_Story": user_story,
        "User_Input": user_input,
        "Expected_Response": expected_response,
        "Actual_Response": None,
        "Evaluation": "Failed due to repeated rate limits",
        "Emotion": emotion
    }

def run_tests_and_save_results(input_csv_path, app):
    # Load test cases from input CSV, ignoring the Notes column
    print(f"Loading test cases from {input_csv_path}...")
    test_cases = pd.read_csv(input_csv_path)
    print(f"Loaded {len(test_cases)} test cases.")

    # Prepare a list to store results
    results = []

    # We use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for _, row in test_cases.iterrows():
            session_id = generate_random_session_id()
            futures.append(executor.submit(process_test_case, row, app, session_id))

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing test case: {e}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Create the output CSV file path
    input_filename = os.path.basename(input_csv_path)
    output_csv_path = f"results_llm_judge_{input_filename}"

    # Save results to a new CSV file
    results_df.to_csv(output_csv_path, index=False)

    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    app = App()  # Initialize AI app
    input_csv_path = "test_dataset.csv"  # Path to the CSV file containing test cases

    # Run the test cases and save the results
    run_tests_and_save_results(input_csv_path, app)

