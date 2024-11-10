import pandas as pd
import argparse

def parse_evaluation(evaluation_text):
    """
    Parses an evaluation string into a dictionary of scores.
    """
    scores = {}
    for line in evaluation_text.splitlines():
        if ':' in line:
            category, score = line.split(':')
            scores[category.strip()] = int(score.strip())
    return scores

def evaluate_llm_judge(output_csv_path, output_results_csv):
    # Load the CSV file
    df = pd.read_csv(output_csv_path)
    
    # Initialize lists to track matching and mismatched scores
    score_columns = ['Clarity', 'Follow-Up', 'Relevance', 'Engagement', 
                     'Empathy', 'Supportiveness', 'Politeness', 'Correction Handling', 'Total Score']
    
    correct_counts = {col: 0 for col in score_columns}
    total_counts = {col: 0 for col in score_columns}

    # Iterate through each row and compare the AI and human evaluations
    for index, row in df.iterrows():
        human_evaluation = parse_evaluation(row['Human_Evaluation'])
        ai_evaluation = parse_evaluation(row['Evaluation'])
        
        # Compare each score category
        for category in score_columns:
            if category in ai_evaluation and category in human_evaluation:
                if ai_evaluation[category] == human_evaluation[category]:
                    correct_counts[category] += 1
                total_counts[category] += 1
    
    # Calculate and display the accuracy for each category
    accuracy = {category: (correct_counts[category] / total_counts[category]) * 100 
                if total_counts[category] > 0 else None for category in score_columns}
    
    accuracy_df = pd.DataFrame(list(accuracy.items()), columns=['Category', 'Accuracy (%)'])
    
    # Print the accuracy results
    print("LLM Judge Evaluation Accuracy by Category:")
    print(accuracy_df)
    
    # Save the accuracy results to a CSV file
    accuracy_df.to_csv(output_results_csv, index=False)
    print(f"Accuracy results saved to {output_results_csv}")
    
    return accuracy_df

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Evaluate LLM judge accuracy against human evaluations.')
    
    # Define command-line arguments
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file containing LLM and human evaluations.')
    output_results_csv = 'accuracy_results.csv'

    # Parse arguments
    args = parser.parse_args()

    # Call the evaluation function with the provided arguments
    evaluate_llm_judge(args.input_csv, output_results_csv)

if __name__ == "__main__":
    main()
