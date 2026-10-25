import re
import nltk
import os
import unicodedata

from cs50 import SQL
from flask import redirect, session
from functools import wraps
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

nltk.download('stopwords')
nltk.download('wordnet')


# Initialise Models
# Pre-trained transformer model for sentiment analysis (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# OpenAI client using API key from environment
client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )

# Lemmatizer model
lemmatizer = WordNetLemmatizer()


# Define helper functions
def analyze_sentiment(text):
    """Create sentiment service function"""
    result = sentiment_pipeline(text)[0]

    return result["label"]

# Regex patterns for feedback tagging
bug_patterns = [
        r"\bcrash(?:es|ed|ing)?\b",
        r"\berror(?:s)?\b",
        r"\bbug(?:s|gy)?\b",
        r"\bglitch(?:es|y)?\b",
        r"\bnot working\b",
        r"\bdoesn['’]?t work\b",
        r"\bdoes not work\b",
        r"\bwon['’]?t work\b",
        r"\bcan['’]?t (?:play|listen|download|open|login|log in)\b",
        r"\bfailed?\b",
        r"\bfailure\b",
        r"\bkeeps? (?:crashing|stopping|freezing|pausing)\b",
        r"\bconstantly (?:stops?|crashes?|freezes?)\b",
        r"\b(stops?|stopped|stopping) playing\b",
        r"\bfreez(?:e|es|ing)\b",
        r"\bdisconnect(?:s|ed|ing)?\b",
        r"\boffline\b",
        r"\bglitch\b",
        r"\bdoesn['’]?t load\b",
        r"\bwon['’]?t load\b",
    ]

feature_patterns = [
    r"\bwould like\b",
    r"\bwould love\b",
    r"\bplease add\b",
    r"\bcould you add\b",
    r"\bcan you add\b",
    r"\bi wish\b",
    r"\bwe wish\b",
    r"\bshould add\b",
    r"\bshould have\b",
    r"\bneed(?:s)? (?:a|an|the)\b",
    r"\bwant(?:s)? (?:a|an|the)\b",
    r"\badd (?:a|an|the)\b",
    r"\bbring back\b",
    r"\bintroduce\b",
    r"\bmake it possible\b",
    r"\bit would be nice\b",
    r"\bi'd like\b",
    r"\bid like\b",
    r"\boption to\b",
    r"\bability to\b",
    r"\ballow (?:me|us) to\b",
]

complaint_patterns = [
    r"\bterrible\b",
    r"\bworst\b",
    r"\bawful\b",
    r"\bhorrible\b",
    r"\bfrustrat(?:ed|ing|ion)\b",
    r"\bannoy(?:ed|ing|ance)\b",
    r"\bdisappoint(?:ed|ing|ment)\b",
    r"\bhate\b",
    r"\bpoor\b",
    r"\bbad\b",
    r"\bexpensive\b",
    r"\btoo expensive\b",
    r"\boverpriced\b",
    r"\bunacceptable\b",
    r"\birritat(?:ed|ing)\b",
    r"\bscam\b",
    r"\bridiculous\b",
    r"\bunhappy\b",
    r"\bdislike\b",
]

praise_patterns = [
    r"\blove\b",
    r"\bloved\b",
    r"\bgreat\b",
    r"\bexcellent\b",
    r"\bamazing\b",
    r"\bawesome\b",
    r"\bfantastic\b",
    r"\bperfect\b",
    r"\bbrilliant\b",
    r"\bhelpful\b",
    r"\beasy to use\b",
    r"\buser[- ]friendly\b",
    r"\bbest\b",
    r"\bnice\b",
    r"\bgood\b",
    r"\bexcellent\b",
    r"\bthank(?:s| you)\b",
    r"\bworks? great\b",
    r"\bworks? perfectly\b",
]

bug_regex = re.compile("|".join(bug_patterns))
feature_regex = re.compile("|".join(feature_patterns))
complaint_regex = re.compile("|".join(complaint_patterns))
praise_regex = re.compile("|".join(praise_patterns))
complaint_compiled = [re.compile(p) for p in complaint_patterns]
praise_compiled = [re.compile(p) for p in praise_patterns]

def assign_tag(text, rating=None):
    """Assign tags to customer feedback."""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()

    if rating is not None and str(rating).strip() != "":
        try:
            rating = float(rating)
        except (ValueError, TypeError):
            rating = None
    else:
        rating = None

    if bug_regex.search(text):
        return "Bug Report"

    if feature_regex.search(text):
        return "Feature Request"

    complaint_match = complaint_regex.search(text)
    praise_match = praise_regex.search(text)

    if complaint_match and not praise_match:
        return "Complaint"

    if praise_match and not complaint_match:
        return "Praise"

    if complaint_match and praise_match:

        complaint_score = sum(
            bool(pattern.search(text))
            for pattern in complaint_compiled
        )

        praise_score = sum(
            bool(pattern.search(text))
            for pattern in praise_compiled
        )

        if complaint_score > praise_score:
            return "Complaint"

        if praise_score > complaint_score:
            return "Praise"

    if rating is not None:
        if rating >= 4:
            return "Praise"
        elif rating <= 2:
            return "Complaint"

    return "Other"

# Stopwords list
STOP_WORDS = list(
        stopwords.words("english")
        + [
            "spotify",
            "app",
            "music",
            "song",
            "best",
            "worst",
            "love",
            "even"
        ]
    )

def extract_topics(feedback, min_clusters=4, max_clusters=8):
    """Implement topic extraction"""
    # Convert feedback to TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.7
    )
    X = vectorizer.fit_transform(feedback)

    # Group similar feedback using KMeans Clustering
    scores = {}
    models = {}

    for k in range(min_clusters, max_clusters + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)
        scores[k] = score
        models[k] = model

    best_k = max(scores, key=scores.get)
    model = models[best_k]

    return model, vectorizer

def generate_ai_insights(feedback):
    system_prompt = f"""
    You are a Product Analyst.
    Your job is to interpret the provided user feedback and analytics.

    Generate:

    1. An executive summary of no more than 100 words.
    2. Three key findings. Each finding should explain an important
    pattern or observation.
    3. Three actionable recommendations based on those findings.

    Return valid JSON in exactly this structure:

    {{
        "summary": "string",
        "issues": [
            {{
                "title": "string",
                "explanation": "string"
                "evidence:: "string"
            }}
        ],
        "recommendations": [
            {{
                "title": "string",
                "explanation": "string",
                "evidence:: "string"
            }}
        ]
    }}
    """

    input = json.dumps(feedback, indent=2)
    response = client.responses.create(
        model="gpt-5.1",
        input=input,
        instructions=system_prompt
    )

    result = json.loads(response.output_text)

    db.execute("INSERT INTO insights (summary, issues, recommendations, user_id) VALUES (?, ?, ?, ?)", result["summary"], json.dumps(result["issues"]), json.dumps(result["recommendations"]), session.get("user_id"))

    return result

def get_topic_keywords(model, vectorizer, top_n=1):
    terms = vectorizer.get_feature_names_out()
    centroids = model.cluster_centers_
    topics = {}

    for cluster_id in range(model.n_clusters):
        ranked_indices = centroids[cluster_id].argsort()[::-1]

        bigrams = [
            terms[i]
            for i in ranked_indices
            if " " in terms[i]
        ]

        unigrams = [
            terms[i]
            for i in ranked_indices
            if " " not in terms[i]
        ]

        selected = bigrams[:top_n]

        if len(selected) < top_n:
            selected += unigrams[:top_n - len(selected)]

        topics[cluster_id] = selected

    return topics

def login_required(f):
    """Decorate routes to require login."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)

    return decorated_function

def preprocess_text(feedback):
    """Clean and normalize feedback text before topic extraction"""
    # Convert to lowercase
    feedback = feedback.lower()

    # Remove spaces, punctuation, numbers, emojis
    feedback = feedback.strip()
    feedback = re.sub(
        r"[^a-z\s]",
        "",
        feedback
    )

    # Tokenize
    words = feedback.split()

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)
