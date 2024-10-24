import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
from app import App  # Import your AI application class from app.py
import time

# Load your OpenAI API key from environment variables
load_dotenv()

chat = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4",  # Specify the model you want to use
    temperature=0  # You can adjust temperature for variability in responses
)

def evaluate_with_openai(user_story, user_input, actual_response):
    """
    Uses OpenAI to evaluate the AI's actual response based on the user story acceptance criteria.
    """
    # Define the prompts for each user story
    prompts = {
        1: f"""Please evaluate the following conversation based on the criteria of detecting ambiguity and asking clarifying follow-up questions. Does the AI response meet these requirements?

User input: "{user_input}"
AI's response: "{actual_response}"

Acceptance Criteria:
- The AI should detect ambiguous or unclear requests and provide clarifying follow-up questions.
- The AI should ask targeted follow-up questions to help narrow down user intent, without making incorrect assumptions.
- The AI should not proceed with actions until the user clarifies their intent.
- The conversation should maintain a natural, human-like flow, with no abrupt shifts in tone or behavior after 5 interactions.

Please respond with a detailed evaluation.""",

        2: f"""Please evaluate the following conversation based on the criteria of generating contextually relevant follow-up questions. Does the AI response meet these requirements?

User input: "{user_input}"
AI's response: "{actual_response}"

Acceptance Criteria:
- The AI should generate contextually relevant follow-up questions based on the user's input.
- The follow-up questions should reflect a clear understanding of the user’s previous statements, enhancing the conversational depth.
- The follow-up questions should prompt further engagement and exploration of topics in more detail.
- The conversation should remain consistent and human-like for at least 5 interactions, with no deviation in tone or curiosity.
- The AI should not ask too many questions.

Please provide your evaluation.""",

        3: f"""Please evaluate the following conversation based on the criteria of detecting emotional cues and responding with supportive language. Does the AI response meet these requirements?

User input: "{user_input}"
AI's response: "{actual_response}"

Acceptance Criteria:
- The AI should accurately detect emotional cues like frustration or sadness.
- The AI should respond with supportive language that acknowledges the user's emotions, using empathetic phrases.
- The AI should offer appropriate follow-up assistance based on the emotional cue (e.g., offering help or a listening ear).
- The AI should ask open-ended questions, allowing the user to express their challenges freely.
- The conversation should maintain a consistent, supportive tone for at least 5 interactions.

Please provide a detailed evaluation.""",

        4: f"""Please evaluate the following conversation based on the criteria of detecting misunderstandings and handling them politely. Does the AI response meet these requirements?

User input: "{user_input}"
AI's response: "{actual_response}"

Acceptance Criteria:
- The AI should detect and acknowledge misunderstandings when the user indicates a correction is needed.
- The AI should apologize politely and ask clarifying questions before proceeding.
- The AI should not repeat the misunderstanding once corrected.
- The AI should maintain a respectful, human-like tone throughout the interaction.

Please provide your evaluation."""
    }

    # Convert the user_story to an integer (since it's a number in your CSV)
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
        return response.content.strip()  # Use `.content` instead of `.text`
    except Exception as e:
        return f"Error: {str(e)}"

def run_tests_and_save_results(input_csv_path, app, session_id):
    # Load test cases from input CSV, ignoring the Notes column
    print(f"Loading test cases from {input_csv_path}...")
    test_cases = pd.read_csv(input_csv_path)
    print(f"Loaded {len(test_cases)} test cases.")

    # Prepare a list to store results
    results = []

    # Run each test case
    for index, row in test_cases.iterrows():
        print(f"Processing test case {index + 1}/{len(test_cases)}...")
        user_story = row['User_Story']
        user_input = row['User_Input']
        expected_response = row['Expected_Response']
        emotion = row.get('Emotion', None)  # Emotion might be optional

        # Get actual response from AI
        print(f"Sending user input to AI: {user_input}")
        actual_response = app.submit_message(user_input, session_id)
        print(f"Received response from AI.")

        # Use OpenAI to evaluate the response based on the user story
        evaluation = evaluate_with_openai(user_story, user_input, actual_response)
        print(f"Evaluation from OpenAI: {evaluation}")

        # Store the result
        results.append({
            "ID": row["ID"],
            "User_Story": user_story,
            "User_Input": user_input,
            "Expected_Response": expected_response,
            "Actual_Response": actual_response,
            "Evaluation": evaluation,
            "Emotion": emotion
        })
        # Pause for 1 minute after every 10 samples
        if (index + 1) % 10 == 0:
            print("Processed 10 samples, sleeping for 1 minute...")
            time.sleep(60)  # Sleep for 60 seconds

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Create the output CSV file path
    input_filename = os.path.basename(input_csv_path)
    output_csv_path = f"results_llm_judge_{input_filename}"

    # Save results to a new CSV file
    results_df.to_csv(output_csv_path, index=False)

    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    app = App()  # Initialize your AI app
    session_id = "test_session_001"  # Replace with actual session ID as needed
    input_csv_path = "test_dataset.csv"  # Path to the CSV file containing test cases

    # Run the test cases and save the results
    run_tests_and_save_results(input_csv_path, app, session_id)

