# Master's Thesis Code Repository - TFM Mai Anh Võ

- **Author:** Mai Anh Võ
- **Institution:** Complutense University of Madrid
- **Defense Date:** July 2025
- **Language:** Spanish
- **Focus:** Advanced NLP analysis of customer sentiment and topics from Spanish energy companies' Google Play reviews and Twitter data (2023-2024)

## 🎯 Project Overview

This repository contains the complete computational implementation of a Master's Thesis analyzing customer perceptions of five major Spanish energy companies: **Iberdrola, Endesa, Naturgy, Repsol, and TotalEnergies**.

The pipeline processes thousands of Google Play app reviews and Twitter posts, applying state-of-the-art NLP techniques including multilingual sentiment analysis, traditional LDA topic modeling, and transformer-based BERTopic with HDBSCAN clustering. Results include temporal trends, company comparisons, sentiment distributions, and hierarchical topic visualizations.

## 🔬 Research Objectives

1. **Data Pipeline**: Build scalable collection of Spanish-language social media data from multiple sources
2. **Sentiment Analysis**: Classify customer emotions using specialized models (nlptown BERT for reviews, RoBERTuito for tweets)
3. **Topic Discovery**: Extract latent themes using both classical (LDA + coherence optimization) and neural (BERTopic + sentence transformers) approaches
4. **Comparative Insights**: Analyze differences across companies and time periods (2023 vs 2024)
5. **Actionable Visualizations**: Generate publication-ready charts, interactive Plotly dashboards, and topic hierarchies

## 📊 Methodology Pipeline

### 1. Data Collection (10k+ documents/company)

```
Google Play Reviews → google-play-scraper
Twitter (X) Posts → Apify Actor
   └─ Monthly batches (2023-2024)
   └─ Custom queries excluding noise (sports, off-topic)
   └─ Companies: Iberdrola(-sports), Endesa(-basketball), etc.
```


### 2. Data Preprocessing

- **Temporal**: Unix timestamps → pandas datetime (ms/s auto-detection)
- **Text Cleaning**: Lowercase, URL/mention/symbol removal, pysentimiento preprocessing
- **Lemmatization**: spaCy `es_core_news_sm` + custom POS filtering (NOUN, PROPN)
- **Quality**: Deduplication, missing value handling


### 3. Exploratory Data Analysis (EDA)

```
Temporal Trends ─┬─ Line plots: comments/year
                 ├─ Evolution: company × year
                 └─ Histograms: word count distributions

Company Shares ──┼─ Pie charts (custom 5-color palette)
                  └─ Stacked sentiment distributions
```


### 4. Sentiment Analysis Pipeline

| Data Source | Model | Labels | Classification Logic |
| :-- | :-- | :-- | :-- |
| **Reviews** | nlptown/bert-base-multilingual-uncased-sentiment | 1-5★ → POS/NEG/NEU | 50% threshold + 15% margin |
| **Tweets** | pysentimiento (RoBERTuito) | POS/NEU/NEG | Spanish-specialized probabilities |

**Output**: POS/NEU dominance with MIXED category for ambiguous cases

### 5. Topic Modeling (Dual Approach)

**LDA (Gensim)**

```
- Coherence optimization: 2-10 topics (CV metric)
- Spanish stopwords + custom (empresa, energía, temporal terms)
- pyLDAvis interactive visualization
- Word clouds per topic
```

**BERTopic (Neural)**

```
Embedding: all-MiniLM-L6-v2 (sentence-transformers)
Clustering: HDBSCAN (min_cluster_size=2500, eom)
Vectorization: CountVectorizer (ngrams 1-2, max_features=5000)
c-TF-IDF: ClassTfidfTransformer (reduce_frequent_words=True)
Visuals: Barcharts, heatmaps, hierarchies → HTML
```


## 📈 Key Findings

| Metric | Observation |
| :-- | :-- |
| **Volume** | Repsol highest (app reviews), Twitter more balanced |
| **Temporal** | Increasing engagement 2023→2024 across companies |
| **Sentiment** | NEU(40-50%) > POS(25-35%) > NEG(15-25%) > MIXED |
| **Topics** | Billing complaints, service quality, app UX, pricing, sustainability |

**Visual Artifacts Generated**:

- `bartopic.html` - BERTopic barcharts (all topics)
- `barhierarchy.html` - Topic hierarchy tree
- Static PNGs: pies, lines, histograms, wordclouds


## 🛠️ Technology Stack

```
Core: pandas, numpy, scikit-learn, gensim
NLP: transformers, spacy(es_core_news_sm), nltk, pysentimiento, textblob
Viz: matplotlib, seaborn, wordcloud, plotly (interactive HTML)
Clustering: hdbscan, bertopic, sentence-transformers, umap-learn
Scraping: apify-client, requests, google-play-scraper
```


## 🚀 Quick Start

```bash
# 1. Clone & Environment
git clone <repo>
cd TFM-energy-analysis
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 2. Install (pinned versions recommended)
pip install -r requirements.txt

# 3. Download models
python -m spacy download es_core_news_sm
python -m nltk.downloader stopwords punkt

# 4. Environment variables
export APIFY_API="your-apify-token"

# 5. Run full pipeline
python TFM_code_summary.py
```

**Generated Files Structure**:

```
TFM/
├── Twitter/
│   ├── [company][year]Tweets.json (raw monthly)
│   └── Tweetses18Ene2025.json (master dataset)
├── bartopic.html (BERTopic dashboard)
├── barhierarchy.html (topic tree)
└── figures/ (PNG exports)
```


## 🔍 Reproducibility Notes

- **Apify Actor**: (Twitter scraper)
- **Rate Limits**: Monthly batching prevents bans
- **Spanish Focus**: All models tuned for es_ES (accents preserved)
- **Memory**: BERTopic embeddings pre-computed (float32)


## 🌟 Contributions \& Novelty

1. **First comparative NLP study** of Spanish energy sector social data
2. **Dual topic modeling** (LDA + BERTopic) with coherence validation
3. **Production-grade pipeline** with error handling, scalable scraping
4. **Rich visualizations** combining static PNG + interactive Plotly HTML

## 🚧 Limitations
- Spanish-only (no Catalan/Basque customer comments)
- No real-time scraping (static 2023-2024 snapshot)
- HDBSCAN outlier handling may merge niche topics


## 🔮 Future Work

1. **Multi-lingual Extension**

```
- Add Catalan/Basque via `ca_core_news_sm`, `eu_core_news_sm`
- Multilingual BERTopic (mBERT/XLM-R embeddings)
```

2. **Real-time Pipeline**

```
- Streaming Twitter via Tweepy/v2 API
- Airflow/Dagster orchestration
- Daily sentiment dashboards (Streamlit/Gradio)
```

3. **Advanced Modeling**

```
- Zero-shot classification (billing/service/product)
- Aspect-Based Sentiment Analysis (ABSA)
- LLM topic interpretation (GPT-4o/Llama3.1)
- LangChain Agentic Pipeline (Parallel Processing): Replace sequential processing with concurrent LLM chains for sentiment, aspect extraction, and topic synthesis.
```

4. **Causal Analysis**

```
- Event detection (price changes, outages)
- Sentiment impact of marketing campaigns
- Synthetic controls for company interventions
```

5. **Deployment**

```
- FastAPI inference server
- Docker + Azure Container Apps
- GPU acceleration
```

## Author

**Mai Anh Võ**
AI Engineer | Data Scientist
- GitHub: [@VoMaiAnh](https://github.com/VoMaiAnh)

***

*Licensed under MIT for academic use. Commercial deployment requires permission.*

