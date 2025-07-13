# Approaches to Measuring Happiness

A comprehensive research project investigating novel computational methodologies for measuring happiness and subjective well-being using Large Language Models (LLMs) and sentiment propagation techniques to analyze Reddit content, validated against traditional self-reported measures.

## 📋 Project Overview

This study explores multiple approaches to automated happiness measurement, comparing the effectiveness of different computational methods for evaluating subjective well-being in social media content. The research employs three distinct methodologies:

1. **LLM-Based Happiness Analysis**: Using OpenAI's GPT-4o-mini model to directly evaluate happiness levels in Reddit comments
2. **Sentiment Propagation (SentProp)**: Implementing graph-based sentiment propagation using Word2Vec embeddings and psycholinguistic seed words
3. **Comparative Validation**: Cross-validating both approaches against users' self-reported subjective well-being (SWB) measures

### Research Objectives

- **Primary Goal**: Evaluate the correlation between computational happiness scores and traditional subjective well-being measures
- **Secondary Goals**: 
  - Develop scalable methodologies for measuring happiness in social media content
  - Compare effectiveness of different AI-based approaches to happiness measurement
  - Assess validity, reliability, and temporal stability of automated happiness assessment
  - Provide insights into digital well-being assessment techniques for large-scale analysis

## 🔬 Methodology

### Data Collection
- **Platform**: Reddit comments and posts from diverse subreddits
- **Survey Data**: Self-reported subjective well-being scores
  - **Q1**: Short-term happiness metric ("How happy are you right now?")
  - **Q2**: Long-term satisfaction metric ("How satisfied are you with your life?")
- **Dataset Size**: 375,947 total entries from 992 unique users
- **Sampling Strategy**: Stratified sampling based on user activity levels
  - Under 25 comments: 10 users
  - 25-99 comments: 10 users  
  - 100-999 comments: 10 users
  - 1000+ comments: 10 users

### Approach 1: LLM-Based Analysis Pipeline
1. **Data Preprocessing**: Filter content posted before survey responses to ensure temporal validity
2. **Happiness Scoring**: Use GPT-4o-mini to rate happiness on a 1-10 scale with systematic prompting
3. **Batch Processing**: Efficient processing with progress tracking and resume capabilities
4. **Aggregation**: Calculate average happiness scores per user
5. **Validation**: Compare LLM scores with self-reported SWB measures

### Approach 2: Sentiment Propagation (SentProp) Method
1. **Text Preprocessing**: Clean and prepare Reddit comments using NLTK
2. **Word2Vec Training**: Train domain-specific word embeddings (300-dimensional vectors)
3. **Seed Word Selection**: Use XANEW psycholinguistic norms to identify high/low sentiment words
4. **Graph Construction**: Build k-nearest neighbor graphs based on word similarity
5. **Sentiment Propagation**: Implement random walk algorithm to propagate sentiment scores
6. **Multi-dimensional Analysis**: Generate Valence, Arousal, and Dominance scores
7. **User Scoring**: Aggregate word-level sentiment scores to user-level happiness measures

### Statistical Analysis
- Pearson correlation analysis between computational scores and SWB measures
- Cross-validation of happiness measurement methodologies
- Temporal analysis of correlation patterns
- Assessment of score consistency and reliability across different time windows

## 📁 Project Structure

```
Approaches-To-Measuring-Happiness/
├── README.md                          # Comprehensive project documentation
├── Code/                              # Source code and analysis notebooks
│   ├── IntroToHappinessSurvey.ipynb  # Exploratory data analysis and survey insights
│   ├── LLM_Score.py                  # Original LLM analysis script
│   ├── LLM_Score_Final.py            # Production LLM scoring with batch processing
│   ├── llm_score.ipynb               # Interactive LLM analysis notebook
│   ├── SentProp_Score-2.ipynb        # Sentiment propagation implementation
│   └── requirements.txt              # Python dependencies
├── data/                             # Datasets and processed results
│   ├── RedditDataUTF-8.csv          # Main dataset with Reddit comments and survey data
│   └── merged_data.csv               # Processed correlation results
├── images/                           # Research figures and visualizations
│   ├── Process Flow for LLM-Based Analysis.png
│   ├── Figure 3.2.1.png - Figure 5.4.b.png    # Research result visualizations
│   └── Matplotlib Chart.png
├── Approaches to Measuring Happiness.pdf        # Full research paper
└── Final Poster CS598.pptx.pdf                 # Conference presentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenAI API key (for LLM analysis)
- Required Python packages (see requirements.txt)
- XANEW psycholinguistic database (for SentProp method)
- Jupyter Notebook/Lab for interactive analysis

### Dependencies

```
openai
python-dotenv
requests
pandas
scipy
numpy
nltk (for text preprocessing)
gensim (for Word2Vec)
matplotlib/seaborn (for visualizations)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RitulSoni/Approaches-To-Measuring-Happiness.git
cd Approaches-To-Measuring-Happiness
```

2. Install dependencies:
```bash
pip install -r Code/requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file in the Code/ directory
echo "OPENAI_API_KEY=your_api_key_here" > Code/.env
```

4. Download NLTK data (required for text preprocessing):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Usage

#### Method 1: LLM-Based Analysis

**Option A: Interactive Analysis**
```bash
cd Code/
jupyter notebook llm_score.ipynb
```

**Option B: Batch Processing Script** (Recommended for large datasets)
```bash
cd Code/
# Configure your API key first
python LLM_Score_Final.py
# Script includes automatic progress tracking and time estimation
```

**Option C: Original Implementation**
```bash
cd Code/
python LLM_Score.py
```

#### Method 2: Sentiment Propagation Analysis

```bash
cd Code/
jupyter notebook SentProp_Score-2.ipynb
```

#### Method 3: Exploratory Data Analysis

```bash
cd Code/
jupyter notebook IntroToHappinessSurvey.ipynb
```

### Key Components

#### LLM Analysis (`LLM_Score_Final.py`)
- **`process_and_update_reddit_data()`**: Main processing function with configurable batch support
- **`get_llm_score()`**: Analyzes individual text using GPT-4o with error handling and returns happiness score (1-10)
- **Temporal filtering**: Automatically filters content to ensure it predates survey responses (`ContentTimestamp <= SurveyTimestamp`)
- **Progress tracking**: Real-time time estimation and processing statistics for large datasets
- **Robust error handling**: Graceful handling of API failures with NaN fallback values
- **Memory efficiency**: Processes data in configurable batches to manage memory usage

#### Sentiment Propagation (`SentProp_Score-2.ipynb`)
- **Word2Vec training**: Domain-specific embedding generation
- **Graph construction**: k-NN similarity networks
- **Sentiment propagation**: Random walk algorithm implementation
- **Multi-dimensional scoring**: Valence, Arousal, Dominance analysis
- **User aggregation**: Content-level to user-level score mapping

#### Data Exploration (`IntroToHappinessSurvey.ipynb`)
- **Dataset overview**: Comprehensive statistics and distributions
- **User activity analysis**: Comment frequency patterns
- **Subreddit analysis**: Platform-specific happiness patterns
- **Survey response analysis**: Q1 vs Q2 comparison and validation

## 📊 Dataset Information

### Reddit Data Structure
- **Username**: Anonymous user identifier
- **PostID**: Unique identifier for each post/comment
- **Type**: Submission or Comment
- **Content**: Reddit comment/post text
- **ContentTimestamp**: Content creation time
- **SurveyTimestamp**: Survey response time
- **TimeDifferenceHours**: Time between content and survey
- **Metadata**: Upvotes, subreddit, parent relationships
- **Survey Data**: Self-reported well-being scores (Q1: 1-10, Q2: 1-10)

### Data Statistics
- **Total entries**: 375,947 rows
- **Unique users**: 992 participants
- **Q1 responses**: 539 users (short-term happiness)
- **Q2 responses**: 453 users (long-term satisfaction)
- **Average content length**: 23.4 words per comment
- **Top subreddits**: AskReddit, teenagers, memes, AskMen

## 📈 Key Findings

### LLM-Based Analysis Results
- **Correlation with self-reported scores**: 0.1563 (p < 0.0001)
- **Statistical significance**: Strong evidence of correlation between computational and self-reported happiness measures
- **Model consistency**: Temperature=0 ensures deterministic outputs across runs
- **Processing efficiency**: Batch processing with real-time progress tracking and time estimation
- **Temporal validity**: Automatic filtering ensures content predates survey responses
- **Robustness**: Error handling maintains data integrity during large-scale processing

### Sentiment Propagation Results
- **Valence-Happiness Correlation**: 
  - Q1: 0.1575 (p < 0.0001)
  - Q2: 0.2191 (p < 0.0001)
- **Arousal-Happiness Correlation**:
  - Q1: -0.0940 (p < 0.0001)
  - Q2: -0.1332 (p < 0.0001)
- **Dominance-Happiness Correlation**:
  - Q1: 0.1996 (p < 0.0001)
  - Q2: 0.2038 (p < 0.0001)

### Cross-Method Validation
- **LLM-SentProp Correlation**: Strong correlation between methods validates approach
- **Temporal stability**: Correlations maintain significance across different time windows
- **Method complementarity**: Different approaches capture distinct aspects of happiness

### Research Contributions
- **Methodological innovation**: Novel application of sentiment propagation to happiness measurement
- **Scalability demonstration**: Automated methods suitable for large-scale social media analysis
- **Validation framework**: Comprehensive comparison against self-reported measures
- **Temporal analysis**: Investigation of happiness measurement stability over time

## 🔧 Technical Implementation

### LLM Configuration
- **Model**: GPT-4o (high-performance model for accurate happiness assessment)
- **Temperature**: 0 (deterministic outputs for reproducibility)
- **Max Tokens**: 1000 (sufficient for numerical responses)
- **Prompt Engineering**: Systematic happiness rating instructions with role-based prompting
- **System Prompt**: "You are an assistant that rates the happiness expressed in a given text on a scale from 1 to 10, where 1 is very unhappy and 10 is very happy. Only provide the numerical score."
- **Batch Processing**: Configurable batch sizes with progress tracking and time estimation

### Sentiment Propagation Parameters
- **Word2Vec**: 300-dimensional vectors, skip-gram architecture
- **k-NN Graph**: k=10 nearest neighbors for word similarity
- **Random Walk**: α=0.85 propagation parameter, 100 iterations
- **Seed Words**: XANEW 90th/10th percentile thresholds

### Processing Optimizations
- **Batch Processing**: Configurable batch sizes (default: 100) for efficient processing
- **Progress Tracking**: Real-time time estimation and completion percentage for large datasets
- **Error Handling**: Robust handling of API failures and data inconsistencies with graceful fallbacks
- **Memory Management**: Efficient processing of large datasets with controlled memory usage
- **Resume Capability**: Processing can be interrupted and resumed from checkpoints

### Performance Metrics
- **Dataset Size**: Successfully processes 375,947+ Reddit entries
- **Processing Speed**: Variable based on API rate limits and batch configuration
- **Error Rate**: Minimal data loss due to robust error handling with NaN fallbacks
- **Memory Efficiency**: Processes large datasets without memory overflow issues

## ⚠️ Limitations

### Technical Limitations
- **API Costs**: Large-scale LLM analysis requires significant OpenAI API usage (consider using GPT-4o-mini for cost optimization)
- **Rate Limits**: Processing speed constrained by API call restrictions (current implementation includes error handling)
- **Context Window**: Short text snippets may lack sufficient context for accurate assessment
- **Model Bias**: LLM may have inherent biases in happiness assessment
- **Error Recovery**: Some scores may be missing due to API failures (handled with NaN values)

### Methodological Limitations
- **Platform Specificity**: Results specific to Reddit user population and culture
- **Temporal Constraints**: Content filtered to precede survey responses
- **Sample Size**: Limited number of users with complete survey responses
- **Self-Selection Bias**: Participants may not represent general population

### Data Limitations
- **Missing Values**: Some LLM scores missing due to API failures
- **Temporal Misalignment**: Variable time gaps between content and survey
- **Content Quality**: Varying quality and length of Reddit comments
- **Survey Validity**: Self-reported measures subject to response bias

## 🔮 Future Work

### Methodological Extensions
- **Multi-platform Validation**: Extend analysis to Twitter, Facebook, Instagram
- **Longitudinal Studies**: Track happiness changes over extended periods
- **Demographic Analysis**: Investigate happiness patterns across user demographics
- **Cultural Validation**: Test approaches across different cultural contexts
- **Real-time Analysis**: Implement streaming processing for live happiness monitoring

### Technical Improvements
- **Model Comparison**: Compare different LLM models (GPT-4, Claude, Llama)
- **Ensemble Methods**: Combine multiple approaches for improved accuracy
- **Real-time Processing**: Develop streaming analysis capabilities
- **Multimodal Analysis**: Incorporate images, videos, and other media types

### Application Domains
- **Mental Health Monitoring**: Clinical applications for depression screening
- **Policy Analysis**: Evaluate impact of social policies on population wellbeing
- **Product Development**: Assess user satisfaction and engagement
- **Social Research**: Large-scale studies of societal happiness trends

## 📊 Code Files Overview

### `IntroToHappinessSurvey.ipynb`
- **Purpose**: Comprehensive exploratory data analysis
- **Key Features**: Dataset statistics, distribution analysis, visualization
- **Outputs**: Data quality assessment, user activity patterns, survey response analysis

### `LLM_Score.py`
- **Purpose**: Original LLM-based happiness scoring implementation
- **Key Features**: Stratified sampling, OpenAI API integration, correlation analysis
- **Outputs**: User-level happiness scores, correlation with SWB measures

### `LLM_Score_Final.py`
- **Purpose**: Production-ready LLM scoring with batch processing
- **Key Features**: Batch processing, progress tracking, temporal filtering
- **Outputs**: Processed dataset with LLM scores, batch processing reports

### `llm_score.ipynb`
- **Purpose**: Interactive notebook for LLM analysis
- **Key Features**: Step-by-step analysis, visualization, documentation
- **Outputs**: Interactive analysis results, statistical summaries

### `SentProp_Score-2.ipynb`
- **Purpose**: Sentiment propagation implementation
- **Key Features**: Word2Vec training, graph construction, multi-dimensional analysis
- **Outputs**: Valence/Arousal/Dominance scores, correlation analysis, temporal validation

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{soni2024approaches,
  title={Approaches to Measuring Happiness: A Large Language Model Analysis of Reddit Content},
  author={Soni, Ritul},
  year={2024},
  journal={CS598 Research Project}
}
```

## 📄 License

This project is available for academic and research purposes. Please respect Reddit's terms of service and OpenAI's usage policies when using this code.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the methodology or extend the analysis.

Areas for contribution:
- Additional sentiment analysis methods
- Improved temporal analysis techniques
- Cross-platform validation studies
- Enhanced visualization and reporting tools

## 📧 Contact

For questions about this research or collaboration opportunities, please open an issue in this repository.

---

**Note**: This research was conducted as part of CS598 coursework investigating computational approaches to happiness measurement. Ensure you have proper API credentials and respect rate limits when running the analysis. The study demonstrates the feasibility of automated happiness assessment but should be validated in specific application contexts. 