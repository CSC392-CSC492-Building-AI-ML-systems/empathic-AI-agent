import pandas as pd
from app import App  # Import the AI application class from app.py
import os

def run_tests_and_save_results(input_csv_path, app, session_id):
    # Load test cases from input CSV
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

        # Determine if the test passed or failed
        status = "Passed" if actual_response.strip().lower() == expected_response.strip().lower() else "Failed"
        
        # Store the result
        results.append({
            "ID": row["ID"],
            "User_Story": user_story,
            "User_Input": user_input,
            "Expected_Response": expected_response,
            "Actual_Response": actual_response,
            "Status": status,
            "Emotion": emotion
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Create the output CSV file path
    input_filename = os.path.basename(input_csv_path)
    output_csv_path = f"results_{input_filename}"
    
    # Save results to a new CSV file
    results_df.to_csv(output_csv_path, index=False)
    
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    app = App()  # Initialize your AI application
    session_id = "test_session"  # You can generate or use a fixed session ID for testing

    # Run tests on both CSV files
    print("Running tests on short inputs...")
    run_tests_and_save_results('test_dataset.csv', app, session_id)

    print("\nRunning tests on long inputs...")
    run_tests_and_save_results('test_dataset_100_samples.csv', app, session_id)

