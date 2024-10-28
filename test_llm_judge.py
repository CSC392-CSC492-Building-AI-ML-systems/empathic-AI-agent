import pandas as pd
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from app import App  # Import AI application class from app.py
from utilities import generate_random_session_id

load_dotenv()

chat = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",
    temperature=0  # You can adjust temperature for variability in responses
)

def evaluate_with_openai(user_story, user_input, actual_response):
    """
    Uses OpenAI to evaluate the AI's actual response based on the user story acceptance criteria.
    """
    # Define the prompts for each user story
    prompts = {
        1: f"""Please evaluate the following conversation based on the criteria of detecting ambiguity and asking clarifying follow-up questions. Score each category from 0 to 3.

User input: "{user_input}"
AI's response: "{actual_response}"

Criteria:
- Clarity: Does the response clearly address the user input without ambiguities? (0: No clarity, 3: Very clear)
- Follow-Up Questions: Does the AI ask targeted follow-up questions to clarify user intent? (0: No follow-up questions, 3: Highly relevant follow-up questions)

Please provide your response in this format:
Clarity: X
Follow-Up Questions: Y
Total Score: Z""",

        2: f"""Please evaluate the following conversation based on the criteria of generating contextually relevant follow-up questions. Score each category from 0 to 3.

User input: "{user_input}"
AI's response: "{actual_response}"

Criteria:
- Relevance: Does the response accurately address the user's input? (0: Off-topic, 3: Fully relevant)
- Engagement: Does the AI generate appropriate follow-up questions to deepen the conversation? (0: No engagement, 3: Highly engaging)

Please provide your response in this format:
Relevance: X
Engagement: Y
Total Score: Z""",

        3: f"""Please evaluate the following conversation based on the criteria of detecting emotional cues and responding with supportive language. Score each category from 0 to 4.

User input: "{user_input}"
AI's response: "{actual_response}"

Criteria:
- Empathy: Does the AI recognize and respond to emotional cues effectively? (0: No recognition, 4: Excellent recognition)
- Supportiveness: Does the AI offer appropriate support and follow-up assistance? (0: No support, 4: Highly supportive)

Please provide your response in this format:
Empathy: X
Supportiveness: Y
Total Score: Z""",

        4: f"""Please evaluate the following conversation based on the criteria of detecting misunderstandings and handling them politely. Score each category from 0 to 3.

User input: "{user_input}"
AI's response: "{actual_response}"

Criteria:
- Politeness: Does the AI maintain a polite and respectful tone throughout the interaction? (0: Rude, 3: Consistently polite)
- Correction Handling: Does the AI acknowledge misunderstandings and respond appropriately? (0: No acknowledgment, 3: Fully acknowledges and clarifies)

Please provide your response in this format:
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

    # Use ChatOpenAI's invoke method to generate a response
    try:
        response = chat.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def parse_scores(evaluation, user_story):
    scores = {}
    try:
        for line in evaluation.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)  # Split on the first colon only
                key = key.strip()
                value = value.strip()
                
                # Ensure value is an integer
                try:
                    scores[key] = int(value)
                except ValueError:
                    continue
        
        # Calculate the total score if it's not explicitly included
        if "Total Score" not in scores:
            scores["Total Score"] = sum(value for key, value in scores.items() if isinstance(value, int))

    except Exception as e:
        print(f"Error parsing scores: {e}")
        return {"Error": str(e)}

    return scores

def process_test_case(row, app, session_id):
    user_story = row['User_Story']
    user_input = row['User_Input']
    expected_response = row['Expected_Response']
    emotion = row.get('Emotion', None)

    actual_response = app.submit_message(user_input, session_id)
    evaluation = evaluate_with_openai(user_story, user_input, actual_response)

    # Parse scores specific to the user story
    scores = parse_scores(evaluation, user_story)

    return {
        "ID": row["ID"],
        "User_Story": user_story,
        "User_Input": user_input,
        "Expected_Response": expected_response,
        "Actual_Response": actual_response,
        "Scores": scores,
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

