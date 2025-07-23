import numpy as np
import pandas as pd
import re
import random
import os
import sys
import time

import torch
import torch.nn.functional as F  # Import F for functional API

from sentence_transformers import SentenceTransformer, util

from fastapi import FastAPI, HTTPException, Query, Body
from typing import List, Dict, Optional

from pydantic import BaseModel

torch.manual_seed(42)

app = FastAPI()


class KeywordRecommendationRequest(BaseModel):
    keyword: str
    num_recommendations: int = 20


class UserRecommendationRequest(BaseModel):
    skillset: str
    num_recommendations: int = 10
    # diversity_factor removed from BaseModel


# --- GLOBAL DATA AND MODEL LOADING (Runs once at startup) ---
jobs_df: pd.DataFrame
users_df: pd.DataFrame
model: SentenceTransformer
job_full_content_embeddings: torch.Tensor  # Renamed for clarity
job_title_only_embeddings: torch.Tensor  # New global variable for title-only embeddings
user_embeddings: torch.Tensor


def clean_text(text):
    """
    Cleans text by converting to lowercase, retaining alphanumeric characters and spaces,
    and normalizing whitespace.
    Modified to retain numbers and allow more characters for better semantic understanding.
    This function directly impacts the quality of embeddings and thus semantic similarity.
    """
    text = str(text).lower()
    # Allow alphanumeric characters (a-z, 0-9) and spaces. This retains numbers.
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.on_event("startup")
async def load_data_and_model():
    """
    Loads pre-processed dataframes, the SentenceTransformer model, and pre-computed embeddings
    into global variables when the FastAPI application starts up.
    Raises HTTPException if any required files are not found or loading fails.
    """
    global jobs_df, users_df, model, job_full_content_embeddings, job_title_only_embeddings, user_embeddings

    script_dir = os.path.dirname(__file__)

    processed_jobs_file_path = os.path.join(script_dir, 'cleaned_jobs_processed.csv')
    processed_users_file_path = os.path.join(script_dir, 'cleaned_users_processed.csv')
    job_full_content_embeddings_path = os.path.join(script_dir, 'job_full_content_embeddings.npy')
    job_title_only_embeddings_path = os.path.join(script_dir, 'job_title_only_embeddings.npy')  # New path
    user_embeddings_path = os.path.join(script_dir, 'user_embeddings.npy')

    start_time = time.time()
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime(start_time))}] Startup: Starting data and model loading process...")

    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Attempting to load pre-processed DataFrames...")

    try:
        load_jobs_start = time.time()
        jobs_df = pd.read_csv(processed_jobs_file_path)
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Pre-processed job data loaded in {time.time() - load_jobs_start:.2f} seconds. Shape: {jobs_df.shape}")
    except FileNotFoundError:
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] ERROR: 'cleaned_jobs_processed.csv' not found. Please run 'generate_embeddings.py' first.")
        raise HTTPException(status_code=500,
                            detail=f"Processed job data file not found at {processed_jobs_file_path}. Please run generate_embeddings.py.")
    except Exception as e:
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] ERROR: An unexpected error occurred while loading processed job data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load processed job data: {e}")

    try:
        load_users_start = time.time()
        users_df = pd.read_csv(processed_users_file_path)
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Pre-processed user profile data loaded in {time.time() - load_users_start:.2f} seconds. Shape: {users_df.shape}")
    except FileNotFoundError:
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] ERROR: 'cleaned_users_processed.csv' not found. Please run 'generate_embeddings.py' first.")
        raise HTTPException(status_code=500,
                            detail=f"Processed user profile data file not found at {processed_users_file_path}. Please run generate_embeddings.py.")
    except Exception as e:
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] ERROR: An unexpected error occurred while loading processed user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load processed user data: {e}")

    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Loading SentenceTransformer model...")
    load_model_start = time.time()
    try:
        # Load the fine-tuned model
        fine_tuned_model_path = os.path.join(script_dir, 'fine_tuned_job_skill_model')
        if os.path.exists(fine_tuned_model_path):
            model = SentenceTransformer(fine_tuned_model_path)
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Fine-tuned model loaded from local path: {fine_tuned_model_path} in {time.time() - load_model_start:.2f} seconds.")
        else:
            # Fallback to base model if fine-tuned model not found (and prompt user to run fine-tuning)
            model_name = 'all-mpnet-base-v2'
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Fine-tuned model not found at {fine_tuned_model_path}.")
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Loading base '{model_name}' model. Please run 'finetune_model.py' to generate the fine-tuned model for better results.")
            model = SentenceTransformer(model_name)
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Base model loaded via default mechanism in {time.time() - load_model_start:.2f} seconds.")

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S', time.localtime())}] ERROR: Failed to load SentenceTransformer model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load SentenceTransformer model: {e}")

    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Attempting to load pre-computed embeddings...")
    try:
        if os.path.exists(job_full_content_embeddings_path) and \
                os.path.exists(job_title_only_embeddings_path) and \
                os.path.exists(user_embeddings_path):  # Check for all three embedding files
            load_embeddings_start = time.time()
            job_full_content_embeddings = torch.tensor(np.load(job_full_content_embeddings_path))
            job_title_only_embeddings = torch.tensor(np.load(job_title_only_embeddings_path))  # Load new embeddings
            user_embeddings = torch.tensor(np.load(user_embeddings_path))
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Pre-computed embeddings loaded in {time.time() - load_embeddings_start:.2f} seconds.")
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Job full content embeddings shape: {job_full_content_embeddings.shape}")
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: Job title only embeddings shape: {job_title_only_embeddings.shape}")  # Print new shape
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] Startup: User embeddings shape: {user_embeddings.shape}")
        else:
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] ERROR: One or more pre-computed embedding files not found. Please run 'generate_embeddings.py' script first.")
            raise HTTPException(status_code=500,
                                detail="Pre-computed embedding files not found. Please run generate_embeddings.py script first.")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S', time.localtime())}] ERROR: Failed to load pre-computed embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load pre-computed embeddings: {e}")

    end_time = time.time()
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime(end_time))}] Startup: FastAPI application startup complete in {end_time - start_time:.2f} seconds. Ready to receive requests.")


# --- Recommendation Function (Keyword-based) ---
def recommend_sbert(job_keyword: str, num_recommendations: int = 20) -> List[Dict]:
    """
    Generates job recommendations for a given keyword based on Cosine Similarity with job embeddings (full content).
    Returns the top N most precise recommendations.
    """
    cleaned_keyword = clean_text(job_keyword)
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_SBERT: Cleaned keyword for embedding: '{cleaned_keyword}'")

    if not cleaned_keyword:
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_SBERT: Keyword '{job_keyword}' resulted in empty cleaned keyword.")
        return []

    try:
        # Semantic understanding comes from the 'model.encode' step, converting text to meaningful embeddings.
        query_embedding = model.encode(cleaned_keyword, convert_to_tensor=True)
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_SBERT: Query embedding generated. Shape: {query_embedding.shape}")

        # Calculate Cosine Similarity scores against full job content embeddings. Higher score means more similar.
        cosine_scores = util.cos_sim(query_embedding, job_full_content_embeddings)[0]  # Use job_full_content_embeddings
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_SBERT: Cosine Scores calculated. Min: {cosine_scores.min():.4f}, Max: {cosine_scores.max():.4f}")

        # Convert to list of (index, score) tuples for sorting
        scores_list = list(enumerate(cosine_scores.cpu().numpy()))

        # Sort jobs by cosine score in descending order (higher score means more relevant)
        sorted_similar_jobs = sorted(scores_list, key=lambda x: x[1], reverse=True)

        recommended_jobs_details = []
        count = 0
        for i, score in sorted_similar_jobs:
            if count >= num_recommendations:
                break

            # Ensure index 'i' is valid before accessing jobs_df.loc
            if i < len(jobs_df):
                recommended_jobs_details.append({
                    'job_id': int(jobs_df.loc[i, 'Job.ID']),
                    'title': jobs_df.loc[i, 'title'],
                    'company': jobs_df.loc[i, 'company'],
                    'location': jobs_df.loc[i, 'location'],
                    'similarity_score': float(round(score, 4))
                })
                count += 1
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_SBERT: Found {len(recommended_jobs_details)} recommendations.")
        return recommended_jobs_details
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S', time.localtime())}] ERROR in recommend_sbert: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating keyword recommendations: {e}")


# --- Recommendation Function (User-based) ---
def recommend_for_user_sbert(user_skillset: str, num_recommendations: int = 10) -> List[Dict]:
    """
    Generates job recommendations for a given user skillset based on its embedding,
    using Cosine Similarity primarily with job title embeddings.
    Returns the top N most precise recommendations.
    """
    cleaned_skillset = clean_text(user_skillset)
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_User_SBERT: Cleaned skillset for embedding: '{cleaned_skillset}'")

    if not cleaned_skillset:
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_User_SBERT: Provided skillset '{user_skillset}' resulted in empty cleaned skillset.")
        return []

    try:
        # Semantic understanding comes from the 'model.encode' step, converting text to meaningful embeddings.
        target_user_embedding = model.encode(cleaned_skillset, convert_to_tensor=True)
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_User_SBERT: User skillset embedding generated. Shape: {target_user_embedding.shape}")

        # Calculate Cosine Similarity scores against job title only embeddings. Higher score means more similar.
        cosine_scores = util.cos_sim(target_user_embedding, job_title_only_embeddings)[
            0]  # Use job_title_only_embeddings
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_User_SBERT: Cosine Scores calculated for skillset. Min: {cosine_scores.min():.4f}, Max: {cosine_scores.max():.4f}")

        job_scores = list(enumerate(cosine_scores.cpu().numpy()))

        # Sort by score in descending order (higher score means more relevant)
        sorted_similar_jobs = sorted(job_scores, key=lambda x: x[1], reverse=True)

        # No dynamic diversity factor; directly take the top N most similar jobs
        final_recommendations_indices = [item[0] for item in sorted_similar_jobs[:num_recommendations]]

        recommended_jobs_details = []
        if not final_recommendations_indices:
            print(
                f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_User_SBERT: No relevant jobs found for the given skillset.")
            return []

        # Retrieve details for each recommended job
        for i in final_recommendations_indices:
            if i < len(jobs_df):
                recommended_jobs_details.append({
                    'Job.ID': int(jobs_df.loc[i, 'Job.ID']),
                    'title': jobs_df.loc[i, 'title'],
                    'company': jobs_df.loc[i, 'company'],
                    'location': jobs_df.loc[i, 'location'],
                    'similarity_score': float(round(cosine_scores[i].item(), 4))
                })
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] Recommend_User_SBERT: Found {len(recommended_jobs_details)} recommendations.")
        return recommended_jobs_details
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S', time.localtime())}] ERROR in recommend_for_user_sbert: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating user recommendations: {e}")


# --- FastAPI Endpoints ---
@app.get("/recommend/keyword/{keyword}", response_model=List[Dict],
         summary="Get job recommendations based on a keyword (GET)")
async def get_keyword_recommendations(
        keyword: str,
        num_recommendations: int = Query(20, ge=1, le=100, description="Number of recommendations to return (1-100)")
):
    """
    Retrieve the top N most precise job recommendations based on a specified keyword.
    """
    print(f"API Request: GET /recommend/keyword/{keyword} (Num Recs: {num_recommendations})")
    recommendations = recommend_sbert(keyword, num_recommendations)
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found for the given keyword.")
    return recommendations


@app.post("/recommend/keyword", response_model=List[Dict], summary="Get job recommendations based on a keyword (POST)")
async def post_keyword_recommendations(
        request: KeywordRecommendationRequest
):
    """
    Retrieve the top N most precise job recommendations based on a specified keyword via a POST request.
    """
    print(
        f"API Request: POST /recommend/keyword (Keyword: '{request.keyword}', Num Recs: {request.num_recommendations})")
    recommendations = recommend_sbert(request.keyword, request.num_recommendations)
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found for the given keyword.")
    return recommendations


@app.get("/recommend/skillset/{skillset}", response_model=List[Dict],
         summary="Get job recommendations for a specific skillset (GET)")
async def get_skillset_recommendations(
        skillset: str,
        num_recommendations: int = Query(10, ge=1, le=50, description="Number of recommendations to return (1-50)")
):
    """
    Retrieve the top N most precise job recommendations for a specific skillset.
    """
    print(
        f"API Request: GET /recommend/skillset/{skillset} (Num Recs: {num_recommendations})")
    recommendations = recommend_for_user_sbert(skillset, num_recommendations)
    if not recommendations:
        raise HTTPException(status_code=404, detail=f"No recommendations available for the provided skillset.")
    return recommendations


@app.post("/recommend/skillset", response_model=List[Dict],
          summary="Get job recommendations for a specific skillset (POST)")
async def post_skillset_recommendations(
        request: UserRecommendationRequest
):
    """
    Retrieve the top N most precise job recommendations for a specific skillset via a POST request.
    """
    print(
        f"API Request: POST /recommend/skillset (Skillset: '{request.skillset}', Num Recs: {request.num_recommendations})")
    recommendations = recommend_for_user_sbert(request.skillset, request.num_recommendations)
    if not recommendations:
        raise HTTPException(status_code=404,
                            detail=f"No recommendations available for the provided skillset.")
    return recommendations

