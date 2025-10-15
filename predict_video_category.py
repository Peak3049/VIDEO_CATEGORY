#!/usr/bin/env python3
"""
Video Category Prediction Script
Use the trained bag-of-words model to predict video categories from transcripts.
"""

import joblib
import re

class VideoCategoryPredictor:
    def __init__(self):
        # Load the trained model and vectorizer
        self.model = joblib.load('video_category_model_svm.pkl')
        self.vectorizer = joblib.load('tfidf_vectorizer.pkl')

    def preprocess_text(self, text):
        """Preprocess transcript text (same as training)"""
        if not text:
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def predict_category(self, transcript):
        """Predict the category of a video based on its transcript"""
        # Preprocess the transcript
        processed_transcript = self.preprocess_text(transcript)

        # Convert to TF-IDF features
        features = self.vectorizer.transform([processed_transcript])

        # Make prediction
        prediction = self.model.predict(features)[0]

        # Get prediction probabilities (handle SVM without probability=True)
        try:
            probabilities = self.model.predict_proba(features)[0]
            class_names = self.model.classes_

            # Get top 3 predictions with probabilities
            top_3_indices = probabilities.argsort()[-3:][::-1]
            top_3_predictions = [
                (class_names[i], probabilities[i])
                for i in top_3_indices
            ]
        except AttributeError:
            # SVM without probability=True doesn't have predict_proba
            # Return just the prediction with dummy probabilities
            top_3_predictions = [(prediction, 1.0)]

        return prediction, top_3_predictions

def main():
    """Main function for interactive prediction"""
    print("=== VIDEO CATEGORY PREDICTOR ===")
    print("Enter a video transcript to predict its category.")
    print("Type 'quit' to exit.\n")

    predictor = VideoCategoryPredictor()

    while True:
        print("-" * 50)
        transcript = input("Enter video transcript: ")

        if transcript.lower() == 'quit':
            break

        if not transcript.strip():
            print("Please enter a valid transcript.")
            continue

        try:
            # Make prediction
            prediction, top_3 = predictor.predict_category(transcript)

            print(f"\nPredicted Category: {prediction}")
            print("Top 3 Predictions:")
            for i, (category, prob) in enumerate(top_3, 1):
                print(".1f")

        except Exception as e:
            print(f"Error making prediction: {e}")

    print("\nThank you for using the Video Category Predictor!")

if __name__ == "__main__":
    main()
