# InsightFlow

Today, companies receive customer feedback from many different channels such as emails, calls, and social media. This feedback is often unstructured and difficult to synthesise efficiently. As such, much of this data is left untapped. A study by the International Data Corporation in 2023 revealed that approximately 90% of enterprise data is unstructured, but only an estimated 0.5% is used. This results in valuable insights being overlooked and teams finding it challenging to reliably identify the most impactful features.

InsightFlow is a customer feedback analytics web application developed as my CS50 Final Project. It explores how Natural Language Processing  (NLP) and Large Language Models (LLM) can be applied to a practical product management problem - transforming large amounts of unstructured customer feedback into actionable product insights. The project brings together my interests in both product management and data analytics.

#### Video Demo: [InsightFlow Demo](https://youtu.be/GWebQAZeDEo)

## Overview of InsightFlow's Architecture

```text
            Customer Feedback
                    │
                    ▼
              CSV Upload
                    │
                    ▼
          Validation / Pre-processing
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      Sentiment   Topic    Tag
      Analysis   Analysis  Classification
          │         │         │
          └─────────┼─────────┘
                    │
                    ▼
        AI-Generated Recommendations
                    │
                    ▼
             SQLite Database
                    │
                    ▼
             Analytics Dashboard
                    │
                    ▼
              Product Decisions
```

## Features

### 1. Feedback Ingestion
Users can import customer feedback via CSV upload on the Sources page. This is subsequently processed and stored in a database table, which can be viewed with pagination and filtering enabled on the Feedback page.

The following NLP techniques are applied to the customer feedback data during processing:

#### 1.1 Sentiment Analysis
InsightFlow uses a pre-trained RoBERTa-base model developed by Cardiff University to detect the sentiment (positive / neutral / negative) of each feedback entry.

[Read the model documentation here](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

#### 1.2 Topic Extraction
InsightFlow uses the following two-step approach to identify recurring themes / product areas across all feedback entries:

**1.2.1 TF-IDF Vectorization:**
Textual customer feedback data is converted into numerical representations based on the importance of the words it contains. Parameters were finetuned including min_df, max_df, and stopword removal.

**1.2.2 K-Means Clustering:**
Similar feedback (based on their TF-IDF vectors) are grouped into k clusters, with each cluster representing a recurring theme / product area. The value of k was determined by optimising the silhouette score.

The most representative terms within each cluster were then used to generate a human-readable topic label for the corresponding feedback entries. Bigrams were prioritised when generating labels to capture more descriptive, meaningful topics.

#### 1.3 Smart Tagging
Each feedback entry is also tagged to one of the following categories: Bug Report, Feature Request, Complaint, Praise, or Other.

Regular expressions are used to identify keywords / patterns in the feedback text. Rule-based classification is then used to assign each entry to the most relevant tag.

### 2. Analytics Dashboard
The Dashboard page provies a consolidated snapshot of customer feedback helping product teams quickly understand what customers are experiencing.

It is also interactive - users can click on chart elements to navigate to a filtered view of the underlying customer feedback. This allows product teams to drill down from high-level trends into the specific feedback driving them, making it easier to investigate issues and understand customer needs.

Key dashboard metrics / visualisations:
- Top Topics
- AI-generated Executive Summary
- Sentiment Breakdown
- Rating Distribution
- Tag Classification
- Total feedback volume
- Recent customer feedback

### 3. AI Recommendations
 The Insights page moves beyond descriptive analytics towards prescriptive analytics, with actionable recommendations for product teams.

InsightFlow connects to the OpenAI API and provides it with the aggregated customer feedback and NLP analysis generated previously, including sentiment, topic, and tag. The LLM then synthesises these inputs into concise, product-focused insights.

Key outputs:
- Executive Summary
- Key Product Issues
- Recommended Actions

## Tech Stack

**Frontend**
- HTML
- CSS
- Bootstrap
- JavaScript
- Chart.js

**Backend**
- Python
- Flask

**Database**
- SQLite

**ML / NLP Libraries**
- scikit-learn
- NLTK
- Hugging Face Transformers

**LLM**
- OpenAI

## What I Learned
Building InsightFlow involved applying concepts across software development, data analysis, NLP, and product management.

Key areas I worked with include:

- Building a Flask web application
- Handling errors and invalid user input
- Natural Language Processing
- Supervised learning
- Unsupervised learning
- Transformers
- Integrating an AI / LLM API
- Designing a SQLite database
- Data Visualisation with Chart.js

## Future Improvements
InsightFlow was designed as a Minimum Viable Product (MVP) to demonstrate how unstructured customer feedback can be transformed into actionable product insights. At the beginning of the project, the Good, Better, and Best goals were defined, with the final submission aligning most closely with the Better outcome.

Potential future improvements include:
- Support for additional feedback sources such as Reddit, email, and support platforms
- PostgreSQL for production-scale data storage
- React frontend for a more dynamic, interactive, and scalable user interface
- Feedback trend analysis
- Custom topics / tags
- Priority scoring
- Conversational Analytics
