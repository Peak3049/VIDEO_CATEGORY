# Video Category Classification

A machine learning project that classifies YouTube video categories based on English transcripts using natural language processing and traditional ML algorithms.

## 📋 Overview

This project implements a text classification system that analyzes YouTube video transcripts to automatically categorize videos into different content categories (e.g., Music, Education, Gaming, News, etc.). The system uses TF-IDF vectorization and SVM classification to achieve video categorization.

## 🚀 Features

- **Multi-dataset Integration**: Combines multiple CSV datasets containing video transcripts and categories
- **Text Preprocessing**: Comprehensive text cleaning and normalization pipeline
- **Multiple ML Algorithms**: Compares Logistic Regression, Random Forest, and SVM models
- **Model Evaluation**: Detailed performance metrics and confusion matrix analysis
- **Interactive Prediction**: Command-line interface for real-time video category prediction
- **Model Persistence**: Saves trained models and vectorizers for production use

## 📊 Dataset

The project uses multiple YouTube transcript datasets:
- `TAP_transcript_en.csv`
- `youtube_data_transcript.csv`
- `youtube_data_transcripted_en_pun.csv`
- `youtube_transcript_en_trained.csv`
- `youtube_transcript_for_trained.csv`

Each dataset should contain at least two columns:
- `transcript_en`: The video transcript text in English
- `category`: The video category label

## 🛠 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

Install the required packages:

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

Or create a requirements.txt file with:

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 📖 Usage

### 1. Training the Model

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
video-category-classification/
│
├── video_category_classifier.py    # Main training pipeline
├── predict_video_category.py       # Prediction interface
├── test_prediction.py             # Model testing script
├── model_analysis.py              # Performance analysis
│
├── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
├── video_category_model_svm.pkl   # Trained SVM model
├── confusion_matrix_svm.png       # Confusion matrix visualization
│
├── TAP_transcript_en.csv          # Training datasets
├── youtube_data_transcript.csv
├── youtube_data_transcripted_en_pun.csv
├── youtube_transcript_en_trained.csv
├── youtube_transcript_for_trained.csv
│
├── __pycache__/                   # Python cache files
└── README.md                      # This file
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

### Machine Learning Models

The pipeline trains and compares three algorithms:
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Ensemble method, handles non-linear relationships
- **SVM (Support Vector Machine)**: Best performer, selected as final model

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

- Built with scikit-learn, pandas, and other open-source ML libraries
- Inspired by YouTube content analysis and categorization challenges
- Thanks to the open-source community for excellent ML tools

---

**Note**: This is a baseline implementation using traditional ML approaches. For production-level accuracy, consider implementing deep learning models and collecting more balanced datasets.
