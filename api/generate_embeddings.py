import numpy as np
import pandas as pd
import re
import os
import torch
from sentence_transformers import SentenceTransformer


def clean_text(text):
    """
    Cleans text by converting to lowercase, retaining alphanumeric characters and spaces,
    and normalizing whitespace.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    script_dir = os.path.dirname(__file__)
    jobs_file_path = os.path.join(script_dir, 'ethiojobs.csv')
    users_file_path = os.path.join(script_dir, 'user_profiles.csv')  # Adjust if your userprofile.csv is elsewhere

    print("Loading raw data...")
    try:
        jobs_df = pd.read_csv(jobs_file_path)
        users_df = pd.read_csv(users_file_path)
        print("Raw data loaded successfully.")
    except FileNotFoundError as e:
        print(
            f"Error loading file: {e}. Make sure 'ethiojobs.csv' and 'user_profiles.csv' are correctly placed in the script directory.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during raw data loading: {e}")
        return

    print("Cleaning and preprocessing text data...")
    # Process Job Data
    jobs_df['Job.ID'] = jobs_df['id'].astype(str).str.strip()
    jobs_df['title'] = jobs_df['title'].astype(str).apply(clean_text)  # Apply clean_text here
    jobs_df['company'] = jobs_df['company'].astype(str).apply(clean_text)  # Apply clean_text here
    jobs_df['location'] = jobs_df['location'].astype(str).apply(clean_text)  # Apply clean_text here
    jobs_df['description'] = jobs_df['description'].astype(str).fillna('').apply(
        clean_text)  # Ensure description is cleaned

    # Create full_text_content for general job embeddings (title weighted)
    jobs_df['full_text_content'] = jobs_df['title'] + ' ' + jobs_df['title'] + ' ' + jobs_df['title'] + ' ' + \
                                   jobs_df['company'] + ' ' + jobs_df['location'] + ' ' + \
                                   jobs_df['description']

    # Process User Profile Data
    users_df['user_id'] = users_df['user_id'].astype(str).str.strip()
    users_df['skillset'] = users_df['skillset'].astype(str).apply(clean_text)
    users_df['industries'] = users_df['industry'].astype(str).apply(clean_text)
    users_df['locations'] = users_df['location'].astype(str).apply(clean_text)

    if 'experience_level' in users_df.columns:
        users_df['experience_level'] = users_df['experience_level'].astype(str).apply(clean_text)
        users_df['user_profile_content_for_embedding'] = users_df['skillset'] + ' ' + \
                                                         users_df['industry'] + ' ' + \
                                                         users_df['location'] + ' ' + \
                                                         users_df['experience_level']
    else:
        users_df['user_profile_content_for_embedding'] = users_df['skillset'] + ' ' + \
                                                         users_df['industry'] + ' ' + \
                                                         users_df['location']

    # Apply cleaning function to the combined user profile content
    users_df['user_profile_content_for_embedding'] = users_df['user_profile_content_for_embedding'].apply(clean_text)

    # Handle missing values *after* text cleaning
    jobs_df.dropna(subset=['id', 'title', 'location', 'description', 'full_text_content'], inplace=True)
    jobs_df.reset_index(drop=True, inplace=True)

    users_df.dropna(subset=['user_id', 'skillset', 'industry', 'location', 'user_profile_content_for_embedding'],
                    inplace=True)
    users_df.reset_index(drop=True, inplace=True)

    if jobs_df.empty or users_df.empty:
        print("Error: DataFrame is empty after cleaning. Cannot generate embeddings or save processed data.")
        return

    print("Loading SentenceTransformer model...")
    try:
        # Load the fine-tuned model
        fine_tuned_model_path = os.path.join(script_dir, 'fine_tuned_job_skill_model')
        if os.path.exists(fine_tuned_model_path):
            model = SentenceTransformer(fine_tuned_model_path)
            print(f"Fine-tuned model loaded from: {fine_tuned_model_path}")
        else:
            # Fallback to base model if fine-tuned model not found (though fine-tuning should be run first)
            print(f"Fine-tuned model not found at {fine_tuned_model_path}. Loading base 'all-mpnet-base-v2' model.")
            model = SentenceTransformer('all-mpnet-base-v2')
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Generating job embeddings (full content - this will take time)...")
    job_full_content_embeddings = model.encode(jobs_df['full_text_content'].tolist(),
                                               convert_to_tensor=True,
                                               show_progress_bar=True,
                                               batch_size=32)
    print("Job full content embeddings generated.")

    print("Generating job title only embeddings (this will take time)...")
    job_title_only_embeddings = model.encode(jobs_df['title'].tolist(),  # Use only the 'title' column
                                             convert_to_tensor=True,
                                             show_progress_bar=True,
                                             batch_size=32)
    print("Job title only embeddings generated.")

    print("Generating user embeddings (this will take time)...")
    user_embeddings = model.encode(users_df['user_profile_content_for_embedding'].tolist(),
                                   convert_to_tensor=True,
                                   show_progress_bar=True)
    print("User embeddings generated.")

    print("Saving processed dataframes and embeddings...")
    # Save processed DataFrames
    jobs_df.to_csv(os.path.join(script_dir, 'cleaned_jobs_processed.csv'), index=False)
    users_df.to_csv(os.path.join(script_dir, 'cleaned_users_processed.csv'), index=False)
    print("Processed DataFrames saved: cleaned_jobs_processed.csv, cleaned_users_processed.csv")

    # Save embeddings
    np.save(os.path.join(script_dir, 'job_full_content_embeddings.npy'), job_full_content_embeddings.cpu().numpy())
    np.save(os.path.join(script_dir, 'job_title_only_embeddings.npy'),
            job_title_only_embeddings.cpu().numpy())  # Save new embeddings
    np.save(os.path.join(script_dir, 'user_embeddings.npy'), user_embeddings.cpu().numpy())
    print("Embeddings saved: job_full_content_embeddings.npy, job_title_only_embeddings.npy, user_embeddings.npy")


if __name__ == "__main__":
    main()
