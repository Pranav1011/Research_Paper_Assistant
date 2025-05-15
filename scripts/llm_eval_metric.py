import pandas as pd

CSV_PATH = 'evaluation_results/llm_eval_results.csv'

def main():
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    correct = (df['judge_response'].str.lower() == 'yes').sum()
    percent = (correct / total) * 100 if total > 0 else 0
    print(f"Total queries evaluated: {total}")
    print(f"Number of correct (yes) answers: {correct}")
    print(f"Accuracy (relevance score): {percent:.2f}%")

if __name__ == "__main__":
    main() 