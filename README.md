# YouTube Video Analysis Pipeline

A comprehensive machine learning project for YouTube video content analysis, featuring end-to-end automated processing from video URLs to category classification.

## 📋 Overview

This project implements a complete pipeline for analyzing YouTube videos:

1. **Data Collection**: Extract video IDs from CSV datasets
2. **Audio Download**: Download audio files using yt-dlp
3. **Speech-to-Text**: Transcribe audio using OpenAI Whisper model
4. **Translation**: Convert transcripts to English using Google Translate API
5. **Text Classification**: Categorize videos using machine learning models

The system uses TF-IDF vectorization and SVM classification for the final categorization step, achieving automated video content analysis.

## 🚀 Features

### Complete Pipeline (`mark_of_youtube_transcript_project.py`)
- **Automated Data Collection**: Extract video IDs from YouTube URLs or CSV datasets
- **Audio Download**: Batch download audio files using yt-dlp with cookie support
- **Speech Recognition**: Convert audio to text using OpenAI Whisper (tiny model)
- **Multi-language Support**: Automatic translation to English using Google Translate API
- **Smart Text Processing**: Chunk-based translation to handle API limits safely

### Classification System
- **Multi-dataset Integration**: Combines multiple CSV datasets containing video transcripts and categories
- **Text Preprocessing**: Comprehensive text cleaning and normalization pipeline
- **Multiple ML Algorithms**: Compares Logistic Regression, Random Forest, and SVM models
- **Model Evaluation**: Detailed performance metrics and confusion matrix analysis
- **Interactive Prediction**: Command-line interface for real-time video category prediction
- **Model Persistence**: Saves trained models and vectorizers for production use

## 📊 Dataset

### Existing Classification Datasets
The project uses multiple YouTube transcript datasets for training:
- `TAP_transcript_en.csv`
- `youtube_data_transcript.csv`
- `youtube_data_transcripted_en_pun.csv`
- `youtube_transcript_en_trained.csv`
- `youtube_transcript_for_trained.csv`

Each dataset should contain at least two columns:
- `transcript_en`: The video transcript text in English
- `category`: The video category label

### Pipeline Input Format
For the complete pipeline (`mark_of_youtube_transcript_project.py`), input should be:
- A CSV file with YouTube video data containing `video_id` or `url` columns
- Optional: Pre-existing `category` column for supervised learning

## 🛠 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

#### For Classification System Only:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

#### For Complete Pipeline (Google Colab):
```bash
pip install yt-dlp transformers librosa soundfile torch deep-translator
```

Or create a requirements.txt file with:

```
# Core ML dependencies
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Complete pipeline dependencies (Colab only)
yt-dlp>=2023.0.0
transformers>=4.20.0
librosa>=0.9.0
soundfile>=0.10.0
torch>=1.12.0
deep-translator>=1.9.0
```

## 📖 Usage

### Complete Pipeline Workflow (`mark_of_youtube_transcript_project.py`)

This Google Colab notebook provides a comprehensive end-to-end solution for YouTube video analysis. **Note**: This script is designed to run in Google Colab environment.

#### Step 1: Data Preparation
- Upload your YouTube video dataset (CSV with video IDs or URLs)
- Extract video IDs from URLs if needed
- Set up Google Drive paths for data storage

#### Step 2: Audio Download
- Configure yt-dlp with cookies for authenticated downloads
- Download audio files in MP3 format
- Batch processing with progress tracking and error handling

#### Step 3: Speech-to-Text Transcription
- Load OpenAI Whisper model (tiny version for efficiency)
- Transcribe audio files to text with timestamps
- Incremental saving to prevent data loss
- Resume capability for interrupted processing

#### Step 4: Translation to English
- Use Google Translate API with chunking for long texts
- Safe processing with rate limiting and error handling
- Automatic language detection and translation

#### Step 5: Video Categorization
- Bag-of-Words vectorization with stop words removal
- Logistic Regression classification
- Performance analysis and visualization
- Word frequency analysis per category

#### Running the Complete Pipeline:
```bash
# This script is designed for Google Colab
# 1. Open in Google Colab
# 2. Mount Google Drive
# 3. Update file paths in the script
# 4. Run cells sequentially
```

### Classification System Usage

#### 1. Training the Model

Run the classification pipeline to train and evaluate models:

```bash
python video_category_classifier.py
```

This will:
- Load and combine all available datasets
- Preprocess the transcript text
- Train multiple ML models (Logistic Regression, Random Forest, SVM)
- Evaluate model performance
- Save the best performing model and vectorizer
- Generate confusion matrix visualizations

### 2. Making Predictions

Use the trained model to predict video categories:

```bash
python predict_video_category.py
```

This launches an interactive interface where you can:
- Input video transcripts
- Get category predictions with confidence scores
- View top 3 predicted categories

### 3. Testing the Model

Run the test script with sample transcripts:

```bash
python test_prediction.py
```

### 4. Analyzing Model Performance

Analyze the trained model's performance and characteristics:

```bash
python model_analysis.py
```

## 📁 Project Structure

```
youtube-analysis-pipeline/
│
├── mark_of_youtube_transcript_project.py  # Complete pipeline (Colab)
├── video_category_classifier.py           # Classification training pipeline
├── predict_video_category.py              # Prediction interface
├── test_prediction.py                     # Model testing script
├── model_analysis.py                      # Performance analysis
│
├── tfidf_vectorizer.pkl                   # Saved TF-IDF vectorizer
├── video_category_model_svm.pkl           # Trained SVM model
├── confusion_matrix_svm.png               # Confusion matrix visualization
│
├── TAP_transcript_en.csv                  # Training datasets
├── youtube_data_transcript.csv
├── youtube_data_transcripted_en_pun.csv
├── youtube_transcript_en_trained.csv
├── youtube_transcript_for_trained.csv
│
├── __pycache__/                           # Python cache files
└── README.md                              # This file
```

## 🤖 Model Details

### Text Preprocessing Pipeline

1. **Lowercasing**: Convert all text to lowercase
2. **Special Character Removal**: Remove punctuation and special characters
3. **Number Removal**: Strip out numeric characters
4. **Whitespace Normalization**: Clean up extra spaces

### Feature Engineering

- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency
- **Feature Limits**: Top 5000 most important features
- **N-gram Range**: Unigrams and bigrams (1-2 words)
- **Document Frequency Filtering**: Terms in 5-80% of documents

### Machine Learning Approaches

#### TF-IDF + SVM Approach (`video_category_classifier.py`)
Trains and compares three algorithms on TF-IDF features:
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Ensemble method, handles non-linear relationships
- **SVM (Support Vector Machine)**: Best performer, selected as final model

#### Bag-of-Words + Logistic Regression (`mark_of_youtube_transcript_project.py`)
- **Count Vectorization**: Simple word frequency counting
- **Logistic Regression**: Efficient classification for text data
- **Stop Words Removal**: Improved feature quality
- **Category-specific Analysis**: Word frequency visualization per category

## 📈 Performance Analysis

### Current Model Performance

- **Overall Accuracy**: ~28.78%
- **Best Algorithm**: SVM with linear kernel
- **Major Challenge**: Significant class imbalance

### Class Distribution (Example)

```
People & Blogs: 44% of dataset
Gaming: 15%
Music: 12%
Education: 8%
News: 5%
[Other categories]: <5% each
```

### Key Insights

1. **Class Imbalance**: "People & Blogs" dominates the dataset
2. **Bias Towards Majority Class**: Model tends to over-predict majority categories
3. **Minority Class Performance**: Poor performance on underrepresented categories
4. **Baseline Performance**: Provides reasonable accuracy for majority classes

## 🔧 Configuration

### Model Hyperparameters

The SVM model uses these default parameters:
- `kernel`: 'linear'
- `probability`: True (for confidence scores)
- `random_state`: 42 (for reproducibility)

### TF-IDF Parameters

- `max_features`: 5000
- `min_df`: 5
- `max_df`: 0.8
- `ngram_range`: (1, 2)

## 🚀 Future Improvements

### Data Enhancement
- Collect more balanced datasets
- Implement data augmentation techniques
- Apply undersampling/oversampling methods

### Advanced Techniques
- **Deep Learning**: BERT, RoBERTa, or other transformer models
- **Word Embeddings**: Word2Vec, GloVe, or contextual embeddings
- **Ensemble Methods**: Combine multiple model predictions

### Feature Engineering
- Include metadata features (video duration, views, etc.)
- Advanced text features (sentiment, topic modeling)
- Multi-modal features (audio, video analysis)

## 🐛 Troubleshooting

### Common Issues

1. **Missing Datasets**: Ensure all CSV files are in the project directory
2. **Memory Errors**: Reduce `max_features` in TF-IDF vectorizer
3. **Import Errors**: Install all required packages
4. **Model Loading**: Ensure `.pkl` files exist before prediction

### Performance Tips

- For faster training: Reduce `max_features` parameter
- For better accuracy: Collect more balanced training data
- For production use: Implement model versioning and monitoring

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 👥 Acknowledgments

- **Machine Learning**: Built with scikit-learn, pandas, and other open-source ML libraries
- **Audio Processing**: yt-dlp for reliable YouTube audio downloads
- **Speech Recognition**: OpenAI Whisper model for accurate transcription
- **Translation**: Google Translate API via deep-translator library
- **Visualization**: Matplotlib and Seaborn for data analysis plots
- **Inspired by**: YouTube content analysis and categorization challenges
- **Thanks to**: Open-source community for excellent ML and NLP tools

## 🔄 Pipeline Options

This project offers two approaches:

1. **Complete Pipeline** (`mark_of_youtube_transcript_project.py`): End-to-end solution from YouTube URLs to categories (Colab-only)
2. **Classification Only** (`video_category_classifier.py`): ML classification using existing transcript data

Choose based on your needs:
- Use the complete pipeline if you have raw YouTube data
- Use classification scripts if you already have transcripts

---

**Note**: This is a baseline implementation using traditional ML approaches. For production-level accuracy, consider implementing deep learning models and collecting more balanced datasets.
