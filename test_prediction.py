#!/usr/bin/env python3
"""
Test script for video category prediction
"""

from predict_video_category import VideoCategoryPredictor

def main():
    predictor = VideoCategoryPredictor()

    # Test transcripts
    test_transcripts = [
        "This is a music video about singing and dancing with lots of rhythm and melody",
        "Today we're going to learn about mathematics and algebra equations",
        "Breaking news: important political developments in the government",
        "Funny comedy skit with jokes and laughter",
        "Gaming walkthrough showing how to beat the final boss",
        "Travel vlog exploring beautiful beaches and tourist destinations"
    ]

    print("=== VIDEO CATEGORY PREDICTION TEST ===\n")

    for i, transcript in enumerate(test_transcripts, 1):
        print(f"Test {i}: {transcript[:50]}...")
        try:
            prediction, top_3 = predictor.predict_category(transcript)
            print(f"Predicted Category: {prediction}")
            print("Top predictions:")
            for category, prob in top_3:
                print(".1f")
            print()
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()

