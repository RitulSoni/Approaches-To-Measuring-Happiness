# Approaches to Measuring Happiness

A research project investigating novel methodologies for measuring happiness and subjective well-being using Large Language Models (LLMs) to analyze Reddit content and validate against traditional self-reported measures.

## 📋 Project Overview

This study explores the effectiveness of using OpenAI's GPT-4o-mini model to evaluate happiness levels in Reddit comments and compares these AI-generated scores with users' self-reported subjective well-being (SWB) measures. The research aims to validate whether LLM-based sentiment analysis can serve as a reliable proxy for measuring happiness at scale.

### Research Objectives

- **Primary Goal**: Evaluate the correlation between LLM-generated happiness scores and traditional subjective well-being measures
- **Secondary Goals**: 
  - Develop a scalable methodology for measuring happiness in social media content
  - Assess the validity and reliability of AI-based happiness measurement
  - Provide insights into digital well-being assessment techniques

## 🔬 Methodology

### Data Collection
- **Platform**: Reddit comments and posts
- **Survey Data**: Self-reported subjective well-being scores (SWB_Q1, SWB_Q2)
- **Sampling Strategy**: Stratified sampling based on user activity levels
  - Under 25 comments: 10 users
  - 25-99 comments: 10 users  
  - 100-999 comments: 10 users
  - 1000+ comments: 10 users

### LLM Analysis Pipeline
1. **Text Preprocessing**: Clean and prepare Reddit comments for analysis
2. **Happiness Scoring**: Use GPT-4o-mini to rate happiness on a 1-10 scale
3. **Aggregation**: Calculate average happiness scores per user
4. **Validation**: Compare LLM scores with self-reported SWB measures

### Statistical Analysis
- Pearson correlation analysis between LLM scores and SWB measures
- Cross-validation of happiness measurement methodologies
- Assessment of score consistency and reliability

## 📁 Project Structure

```
Approaches-To-Measuring-Happiness/
├── README.md                          # Project documentation
├── Code/                              # Source code and analysis
│   ├── LLM_Score.py                  # Main analysis script
│   ├── llm_score.ipynb               # Jupyter notebook implementation
│   └── requirements.txt              # Python dependencies
├── data/                             # Datasets
│   ├── RedditDataUTF-8.csv          # Reddit comments and survey data
│   └── merged_data.csv               # Processed results
├── images/                           # Figures and visualizations
│   ├── Process Flow for LLM-Based Analysis.png
│   ├── Figure 3.2.1.png - Figure 5.4.b.png    # Research figures
│   └── Matplotlib Chart.png
├── Approaches to Measuring Happiness.pdf        # Research paper
└── Final Poster CS598.pptx.pdf                 # Conference poster
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenAI API key
- Required Python packages (see requirements.txt)

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

### Usage

#### Option 1: Run the Python Script
```bash
cd Code/
python LLM_Score.py
```

#### Option 2: Use the Jupyter Notebook
```bash
cd Code/
jupyter notebook llm_score.ipynb
```

### Key Functions

- **`get_happiness_score(text)`**: Analyzes text using GPT-4o-mini and returns happiness score (1-10)
- **`assign_group(count)`**: Categorizes users based on comment frequency
- **Statistical analysis**: Computes correlations between LLM scores and SWB measures

## 📊 Dataset Information

### Reddit Data Structure
- **User ID**: Anonymous user identifier
- **Content**: Reddit comments/posts text
- **Timestamp**: Content creation time
- **Metadata**: Upvotes, subreddit, content type
- **Survey Data**: Self-reported well-being scores (Q1, Q2)

### Sample Data Processing
The analysis focuses on a stratified sample of users to ensure balanced representation across different activity levels while managing computational costs and API limitations.

## 📈 Key Findings

The research demonstrates the potential for LLM-based happiness measurement as a scalable alternative to traditional survey methods, with specific insights into:

- Correlation patterns between AI-generated and self-reported happiness scores
- Effectiveness of different prompting strategies for sentiment analysis
- Scalability considerations for large-scale happiness measurement
- Limitations and biases in automated happiness assessment

*Detailed results and statistical analyses are available in the research paper.*

## 🔧 Technical Implementation

### LLM Configuration
- **Model**: GPT-4o-mini
- **Temperature**: 0 (deterministic outputs)
- **Max Tokens**: 5
- **Prompt**: Systematic happiness rating instruction

### Rate Limiting
- 1-second delay between API calls to respect OpenAI rate limits
- Error handling for failed API requests
- Batch processing capabilities for large datasets

## ⚠️ Limitations

- **API Costs**: Large-scale analysis requires significant OpenAI API usage
- **Rate Limits**: Processing speed limited by API call restrictions  
- **Bias Considerations**: LLM may have inherent biases in happiness assessment
- **Context Limitations**: Short text snippets may lack sufficient context
- **Generalizability**: Results specific to Reddit user population

## 🔮 Future Work

- **Multi-platform Validation**: Extend analysis to other social media platforms
- **Temporal Analysis**: Investigate happiness patterns over time
- **Demographic Studies**: Analyze happiness patterns across user demographics
- **Model Comparison**: Compare different LLM models for happiness assessment
- **Real-time Applications**: Develop real-time happiness monitoring systems

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

## 📧 Contact

For questions about this research or collaboration opportunities, please open an issue in this repository.

---

**Note**: This research was conducted as part of CS598 coursework. Ensure you have proper API credentials and respect rate limits when running the analysis. 