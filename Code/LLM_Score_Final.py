import pandas as pd
import numpy as np
import openai
import time

def process_and_update_reddit_data(api_key, data_file, batch_size=100):
    # Set the OpenAI API key
    openai.api_key = api_key

    # Load the data
    df = pd.read_csv(data_file)

    # Drop rows where ContentTimestamp occurs after SurveyTimestamp
    df_filtered = df[df['ContentTimestamp'] <= df['SurveyTimestamp']].copy()

    # Sort rows by TimeDifferenceHours
    df_sorted = df_filtered.sort_values(by='TimeDifferenceHours')

    # Function to calculate LLM score
    def get_llm_score(text):
        messages = [
            {"role": "system", "content": "You are an assistant that rates the happiness expressed in a given text on a scale from 1 to 10, where 1 is very unhappy and 10 is very happy. Only provide the numerical score."},
            {"role": "user", "content": text}
        ]
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0
            )
            score_str = response.choices[0].message.content
            score = float(score_str)
            return score
        except Exception as e:
            print(f"Error processing text: {e}")
            return np.nan

    # Process one batch and update CSV
    llm_scores = []
    processed_count = 0
    start_time = time.time()

    for start in range(0, df_sorted.shape[0], batch_size):
        batch = df_sorted.iloc[start:start + batch_size]
        
        if batch_size > 1000:
            print("Starting batch processing with time estimation.")

        for idx, row in batch.iterrows():
            if pd.notna(row.get('LlmScore', np.nan)):
                continue  # Skip if LlmScore already present

            text = row['Content']
            score = get_llm_score(text)
            llm_scores.append((idx, score))
            processed_count += 1
            print(f"Processed row {idx}")

            # Update timer every 100 entries if conditions are met
            if batch_size > 1000 and processed_count % 100 == 0:
                elapsed_time = time.time() - start_time
                entries_remaining = batch_size - processed_count
                estimated_total_time = (elapsed_time / processed_count) * entries_remaining
                print(f"Estimated time remaining: {estimated_total_time:.2f} seconds")

            if processed_count >= batch_size:
                break

        # Update the CSV file after processing the batch
        for idx, score in llm_scores:
            df_sorted.at[idx, 'LlmScore'] = score

        df_sorted.to_csv(data_file, index=False)
        print(f"Batch of {batch_size} rows processed and file updated.")
        break  # Stop after processing one batch

# Example usage:
process_and_update_reddit_data(api_key='YOURAPIKEY', data_file='RedditDataUTF-8.csv', batch_size=121242)