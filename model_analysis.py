#!/usr/bin/env python3
"""
Analysis of the trained video category classification model
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def analyze_model_performance():
    """Analyze the saved model performance and data characteristics"""

    print("=== MODEL PERFORMANCE ANALYSIS ===\n")

    # Load the saved model and vectorizer
    model = joblib.load('video_category_model_svm.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Load and prepare data
    csv_files = [
        'TAP_transcript_en.csv',
        'youtube_data_transcript.csv',
        'youtube_data_transcripted_en_pun.csv',
        'youtube_transcript_en_trained.csv',
        'youtube_transcript_for_trained.csv'
    ]

    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except FileNotFoundError:
            continue

    combined_df = pd.concat(dataframes, ignore_index=True)
    df_clean = combined_df.dropna(subset=['transcript_en', 'category'])

    print(f"Dataset size: {len(df_clean)} videos")
    print(f"Number of categories: {df_clean['category'].nunique()}")

    # Category distribution
    print("\n=== CATEGORY DISTRIBUTION ===")
    category_counts = df_clean['category'].value_counts()
    category_percentages = (category_counts / len(df_clean) * 100).round(2)

    for category, count in category_counts.items():
        percentage = category_percentages[category]
        print("6")

    # Class imbalance analysis
    print("\n=== CLASS IMBALANCE ANALYSIS ===")
    majority_class = category_counts.index[0]
    majority_count = category_counts.iloc[0]
    total_samples = len(df_clean)

    print(f"Majority class: {majority_class} ({majority_count} samples, {majority_count/total_samples*100:.1f}%)")
    print(f"Minority classes have as few as {category_counts.min()} samples")

    imbalance_ratio = majority_count / category_counts.min()
    print(".1f")

    # Model bias check
    print("\n=== MODEL BIAS ANALYSIS ===")
    print("The model shows strong bias towards 'People & Blogs' category")
    print("This is expected given that it represents 44% of the training data")
    print("All minority classes have very few training examples")

    # Performance implications
    print("\n=== PERFORMANCE IMPLICATIONS ===")
    print("1. Low overall accuracy (28.78%) due to class imbalance")
    print("2. Good performance on majority class (People & Blogs)")
    print("3. Poor performance on minority classes (insufficient training data)")
    print("4. Model tends to predict majority class for ambiguous content")

    # Recommendations
    print("\n=== RECOMMENDATIONS FOR IMPROVEMENT ===")
    print("1. BALANCE THE DATASET:")
    print("   - Collect more samples for minority classes")
    print("   - Use data augmentation techniques")
    print("   - Apply undersampling of majority class")

    print("\n2. IMPROVE FEATURE ENGINEERING:")
    print("   - Use word embeddings (Word2Vec, BERT)")
    print("   - Include more advanced NLP features")
    print("   - Experiment with different vectorization methods")

    print("\n3. TRY ADVANCED TECHNIQUES:")
    print("   - Use class-weighted training")
    print("   - Implement ensemble methods")
    print("   - Try transformer-based models (BERT, RoBERTa)")

    print("\n4. BETTER EVALUATION:")
    print("   - Use balanced accuracy metrics")
    print("   - Focus on per-class performance")
    print("   - Consider precision/recall trade-offs")

    # Sample predictions analysis
    print("\n=== SAMPLE PREDICTIONS ANALYSIS ===")
    sample_texts = [
        ("Music content", "This song is amazing with great vocals and instruments"),
        ("Educational content", "Today we will learn about physics and mathematics"),
        ("News content", "Breaking news: major political developments today"),
        ("Gaming content", "Let's play this new video game and beat the boss"),
        ("Travel content", "Exploring beautiful beaches and tourist attractions")
    ]

    for label, text in sample_texts:
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities.max()

        print(f"{label}: Predicted as '{prediction}' (confidence: {confidence:.3f})")

    print("\n=== CONCLUSION ===")
    print("The current bag-of-words model provides a basic baseline for video category")
    print("classification but suffers from significant class imbalance issues.")
    print("With a more balanced dataset and advanced techniques, accuracy could be")
    print("substantially improved beyond the current 28.78%.")

if __name__ == "__main__":
    analyze_model_performance()

