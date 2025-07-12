import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import os
import time
import random

# Load your OpenAI API key from the .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Step 1: Load and Preprocess the Data

# Replace 'data.xlsx' with the path to your dataset file
data = pd.read_excel('test.xlsx')

# Ensure that all necessary columns are present
required_columns = ['a', 'Type', 'ContentTimestamp', 'Content', 'Upvotes',
                    'ContentSubreddit', 'Score', 'QuestionID', 'SurveyResponse', 'SurveyTimestamp']

if not all(column in data.columns for column in required_columns):
    raise ValueError("One or more required columns are missing in the dataset.")

# Count the number of comments per user
user_comment_counts = data.groupby('a')['Content'].count().reset_index()
user_comment_counts.columns = ['user', 'comment_count']

# Define user groups based on comment counts
def assign_group(count):
    if count < 25:
        return 'Under 25'
    elif count < 100:
        return '25-99'
    elif count < 1000:
        return '100-999'
    else:
        return '1000+'

user_comment_counts['group'] = user_comment_counts['comment_count'].apply(assign_group)

# Sample users from each group
sampled_users = []
group_sizes = {'Under 25': 10, '25-99': 10, '100-999': 10, '1000+': 10}

for group, size in group_sizes.items():
    users_in_group = user_comment_counts[user_comment_counts['group'] == group]
    sample_size = min(size, len(users_in_group))
    sampled = users_in_group.sample(n=sample_size, random_state=42)
    sampled_users.extend(sampled['user'].tolist())








# Filter the data for the sampled users
sampled_data = data[data['a'].isin(sampled_users)].copy()

# Function to get happiness score from the LLM
def get_happiness_score(text):
    # Prepare the prompt
    messages = [
        {"role": "system", "content": "You are an assistant that rates the happiness expressed in a given text on a scale from 1 to 10, where 1 is very unhappy and 10 is very happy. Only provide the numerical score."},
        {"role": "user", "content": text}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=5,
            n=1,
            stop=None,
            temperature=0
        )
        # Extract the score from the response
        score_str = response['choices'][0]['message']['content'].strip()
        # Convert to float
        score = float(score_str)
        return score
    except Exception as e:
        print(f"Error processing text: {e}")
        return np.nan  # Use NaN for missing values

# Apply the function to the comments
# Since API calls can be rate-limited, we should process with care
happiness_scores = []
for idx, row in sampled_data.iterrows():
    text = row['Content']
    score = get_happiness_score(text)
    happiness_scores.append(score)
    # To avoid hitting the rate limit, add a delay
    time.sleep(1)  # Adjust based on your API rate limits

sampled_data['happiness_score'] = happiness_scores








# Compute average happiness scores per user
user_happiness = sampled_data.groupby('a')['happiness_score'].mean().reset_index()
user_happiness.columns = ['user', 'average_happiness_score']

# Get SWB scores for the sampled users
swb_data = data[data['a'].isin(sampled_users) & data['QuestionID'].isin(['Q1', 'Q2'])]
user_swb = swb_data.groupby(['a', 'QuestionID'])['Score'].mean().reset_index()
user_swb = user_swb.pivot(index='a', columns='QuestionID', values='Score').reset_index()
user_swb.columns = ['user', 'SWB_Q1', 'SWB_Q2']

# Merge happiness scores with SWB scores
merged_data = pd.merge(user_happiness, user_swb, on='user')

# Perform statistical analysis
# For example, compute Pearson correlation between average happiness score and SWB scores
from scipy.stats import pearsonr

# Correlation with SWB_Q1
corr_q1, p_value_q1 = pearsonr(merged_data['average_happiness_score'], merged_data['SWB_Q1'])
print(f"Correlation between LLM happiness scores and SWB_Q1: {corr_q1}, p-value: {p_value_q1}")

# Correlation with SWB_Q2
corr_q2, p_value_q2 = pearsonr(merged_data['average_happiness_score'], merged_data['SWB_Q2'])
print(f"Correlation between LLM happiness scores and SWB_Q2: {corr_q2}, p-value: {p_value_q2}")

# Optional: Save the merged data for further analysis
merged_data.to_csv('merged_data.csv', index=False)