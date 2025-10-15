#!/usr/bin/env python3
"""
Video Category Classification using Bag-of-Words
Trains a model to classify YouTube video categories based on English transcripts.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class VideoCategoryClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.category_mapping = None

    def load_data(self):
        """Load and combine all CSV datasets"""
        csv_files = [
            'TAP_transcript_en.csv',
            'youtube_data_transcript.csv',
            'youtube_data_transcripted_en_pun.csv',
            'youtube_transcript_en_trained.csv',
            'youtube_transcript_for_trained.csv'
        ]

        dataframes = []
        for file in csv_files:
            if os.path.exists(file):
                print(f"Loading {file}...")
                df = pd.read_csv(file)
                dataframes.append(df)
                print(f"  Loaded {len(df)} records from {file}")
            else:
                print(f"Warning: {file} not found")

        if not dataframes:
            raise ValueError("No data files found!")

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Total combined records: {len(combined_df)}")

        return combined_df

    def preprocess_text(self, text):
        """Preprocess transcript text"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def analyze_data(self, df):
        """Analyze the dataset structure and categories"""
        print("\n=== DATA ANALYSIS ===")

        # Check for required columns
        required_cols = ['transcript_en', 'category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Category distribution
        print(f"Total videos: {len(df)}")
        print(f"Categories: {df['category'].nunique()}")
        print("\nCategory distribution:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} videos")

        # Check for missing data
        missing_transcripts = df['transcript_en'].isna().sum()
        missing_categories = df['category'].isna().sum()
        print(f"\nMissing transcripts: {missing_transcripts}")
        print(f"Missing categories: {missing_categories}")

        # Filter out rows with missing data
        df_clean = df.dropna(subset=['transcript_en', 'category'])
        print(f"Records after removing missing data: {len(df_clean)}")

        return df_clean

    def prepare_features_and_labels(self, df):
        """Prepare features and labels for training"""
        # Preprocess transcripts
        print("\nPreprocessing transcripts...")
        df['processed_transcript'] = df['transcript_en'].apply(self.preprocess_text)

        # Create features using TF-IDF
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit to top 5000 features
            min_df=5,           # Ignore terms that appear in less than 5 documents
            max_df=0.8,         # Ignore terms that appear in more than 80% of documents
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )

        X = self.vectorizer.fit_transform(df['processed_transcript'])
        y = df['category']

        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")

        return X, y

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate multiple models"""
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, kernel='linear', probability=True)
        }

        results = {}

        for name, model in models.items():
            print(f"\n=== Training {name} ===")

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'report': report,
                'predictions': y_pred
            }

            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))

        return results

    def plot_confusion_matrix(self, y_test, y_pred, model_name, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred, labels=class_names)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

    def save_model(self, model, model_name):
        """Save the trained model and vectorizer"""
        print(f"\nSaving {model_name} model...")

        # Save model
        joblib.dump(model, f'video_category_model_{model_name.lower().replace(" ", "_")}.pkl')

        # Save vectorizer
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')

        print("Model and vectorizer saved successfully!")

    def run_classification_pipeline(self):
        """Run the complete classification pipeline"""
        print("=== VIDEO CATEGORY CLASSIFICATION PIPELINE ===")

        # Load and combine data
        df = self.load_data()

        # Analyze data
        df = self.analyze_data(df)

        # Prepare features and labels
        X, y = self.prepare_features_and_labels(df)

        # Split data
        print("\nSplitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Train and evaluate models
        results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)

        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_model = results[best_model_name]['model']
        best_accuracy = results[best_model_name]['accuracy']

        print(f"\n=== BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f}) ===")

        # Save best model
        self.save_model(best_model, best_model_name)

        # Plot confusion matrix for best model
        class_names = sorted(y_test.unique())
        self.plot_confusion_matrix(
            y_test,
            results[best_model_name]['predictions'],
            best_model_name,
            class_names
        )

        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")

        return results

def main():
    """Main function"""
    classifier = VideoCategoryClassifier()
    results = classifier.run_classification_pipeline()

    # Print summary
    print("\n=== MODEL COMPARISON SUMMARY ===")
    for model_name, result in results.items():
        print(f"{model_name}: Accuracy = {result['accuracy']:.4f}")

if __name__ == "__main__":
    main()
