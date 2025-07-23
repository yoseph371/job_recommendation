import pandas as pd
import re
import os
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random


# --- Text Cleaning Function (Consistent with main2.py) ---
def clean_text(text):
    """
    Cleans text by converting to lowercase, retaining alphanumeric characters and spaces,
    and normalizing whitespace.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Main Fine-Tuning Logic ---
def main():
    script_dir = os.path.dirname(__file__)
    jobs_file_path = os.path.join(script_dir, 'ethiojobs.csv')
    users_file_path = os.path.join(script_dir, 'user_profiles.csv')

    print("Loading raw data for fine-tuning...")
    try:
        jobs_df = pd.read_csv(jobs_file_path)
        users_df = pd.read_csv(users_file_path)
        print("Raw data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Make sure 'ethiojobs.csv' and 'user_profiles.csv' are correctly placed.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during raw data loading: {e}")
        return

    print("Cleaning and preparing text data for fine-tuning...")
    # Apply cleaning function to relevant columns
    jobs_df['title'] = jobs_df['title'].astype(str).apply(clean_text)
    jobs_df['description'] = jobs_df['description'].astype(str).fillna('').apply(clean_text)
    jobs_df['job industry'] = jobs_df['job industry'].astype(str).apply(clean_text)  # Clean industry
    jobs_df['location'] = jobs_df['location'].astype(str).apply(clean_text)  # Clean location

    users_df['skillset'] = users_df['skillset'].astype(str).apply(clean_text)
    users_df['industry'] = users_df['industry'].astype(str).apply(clean_text)
    users_df['location'] = users_df['location'].astype(str).apply(clean_text)

    # Handle missing values after cleaning
    jobs_df.dropna(subset=['title', 'description', 'job industry', 'location'], inplace=True)
    users_df.dropna(subset=['skillset', 'industry', 'location'], inplace=True)

    if jobs_df.empty or users_df.empty:
        print("Error: DataFrame is empty after cleaning. Cannot prepare data for fine-tuning.")
        return

    # --- Create InputExamples for Fine-Tuning ---
    train_examples = []

    # Max number of examples to generate per user/job category to manage dataset size
    max_positive_matches_per_user = 5
    max_negative_matches_per_user = 5

    # 1. Positive Examples: User Profile vs. Matching Job Titles (by industry/location)
    # This is the primary source of positive examples, focusing on job titles.
    print("Generating positive examples (User Profile vs. Matching Job Titles)...")
    positive_user_job_count = 0
    for _, user_row in users_df.iterrows():
        user_profile_text = f"{user_row['skillset']} {user_row['industry']} {user_row['location']}"
        user_profile_text = clean_text(user_profile_text)

        if user_profile_text:
            # Find jobs matching user's industry OR location
            matching_jobs = jobs_df[
                (jobs_df['job industry'] == user_row['industry']) |
                (jobs_df['location'] == user_row['location'])
                ]

            if not matching_jobs.empty:
                # Sample up to max_positive_matches_per_user matching job titles
                sampled_job_titles = matching_jobs['title'].sample(
                    min(max_positive_matches_per_user, len(matching_jobs))).tolist()
                for job_title in sampled_job_titles:
                    train_examples.append(
                        InputExample(texts=[user_profile_text, job_title], label=1.0))  # Label 1.0 for high similarity
                    positive_user_job_count += 1
    print(f"Added positive examples from User Profile vs. Matching Job Titles: {positive_user_job_count}")

    # 2. Negative Examples: User Profile vs. Non-Matching Job Titles (by industry/location)
    print("Generating negative examples (User Profile vs. Non-Matching Job Titles)...")
    negative_user_job_count = 0
    for _, user_row in users_df.iterrows():
        user_profile_text = f"{user_row['skillset']} {user_row['industry']} {user_row['location']}"
        user_profile_text = clean_text(user_profile_text)

        if user_profile_text:
            # Find jobs NOT matching user's industry AND NOT matching user's location
            non_matching_jobs = jobs_df[
                (jobs_df['job industry'] != user_row['industry']) &
                (jobs_df['location'] != user_row['location'])
                ]

            if not non_matching_jobs.empty:
                # Sample up to max_negative_matches_per_user non-matching job titles
                sampled_jobs = non_matching_jobs['title'].sample(
                    min(max_negative_matches_per_user, len(non_matching_jobs))).tolist()
                for job_title in sampled_jobs:
                    train_examples.append(
                        InputExample(texts=[user_profile_text, job_title], label=0.0))  # Label 0.0 for dissimilarity
                    negative_user_job_count += 1
    print(f"Added negative examples from User Profile vs. Non-Matching Job Titles: {negative_user_job_count}")

    if not train_examples:
        print("No training examples generated. Fine-tuning aborted.")
        return

    print(f"Total generated {len(train_examples)} training examples for fine-tuning.")

    # --- Model Loading and Fine-Tuning ---
    model_name = 'all-mpnet-base-v2'
    output_path = os.path.join(script_dir, 'fine_tuned_job_skill_model')

    print(f"Loading base SentenceTransformer model '{model_name}' for fine-tuning...")
    try:
        local_model_path = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'sentence_transformers',
                                        f'sentence-transformers_{model_name}')

        if os.path.exists(local_model_path):
            print(f"Attempting to load SentenceTransformer model from local path: {local_model_path}")
            model = SentenceTransformer(local_model_path)
            print("Base model loaded from local path.")
        else:
            print(f"Local model not found at {local_model_path}.")
            print(
                f"Attempting to load '{model_name}' via default SentenceTransformer mechanism (will download if not cached elsewhere).")
            model = SentenceTransformer(model_name)
            print("Base model loaded (possibly downloaded).")

    except Exception as e:
        print(f"Error loading base model: {e}. Please ensure you have internet access or the model is cached locally.")
        return

    # Define the DataLoader - explicitly set pin_memory=False
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16, pin_memory=False)

    # Define the loss function (CosineSimilarityLoss is suitable for semantic similarity tasks)
    train_loss = losses.CosineSimilarityLoss(model)

    print("Starting fine-tuning process (this will take time)...")
    num_epochs = 2
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=output_path,
              show_progress_bar=True,
              save_best_model=True,
              optimizer_params={'lr': 2e-5}
              )

    print(f"Fine-tuning complete. Fine-tuned model saved to: {output_path}")


if __name__ == "__main__":
    main()
