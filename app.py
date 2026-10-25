import csv
import io
import json
import os

from cs50 import SQL
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import analyze_sentiment, assign_tag, extract_topics, generate_ai_insights, get_topic_keywords, login_required, preprocess_text

load_dotenv()

# Configure application
app = Flask(__name__)
app.secret_key = os.environ["SECRET_KEY"]

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = ".flask_session_data"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///project.db")


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.errorhandler(Exception)
def handle_error(e):
    code = e.code if isinstance(e, HTTPException) else 500
    return render_template("error.html", error=e), code


@app.route("/")
@login_required
def index():
    """Analytics Dashboard"""
    # Top Topics
    topic_counts = db.execute("SELECT topic, COUNT(*) AS count FROM feedback WHERE user_id = ? GROUP BY topic ORDER BY count DESC LIMIT 5", session.get("user_id"))

    # AI Insights Summary
    llm_output = db.execute("SELECT * FROM insights WHERE user_id = ? ORDER BY generated_at DESC LIMIT 1", session.get("user_id"))

    if llm_output:
        summary = llm_output[0]["summary"]
    else:
        summary = "No AI insights have been generated yet."

    # Sentiment Breakdown
    sentiment_counts = db.execute("SELECT sentiment, COUNT(*) AS count FROM feedback WHERE user_id = ? GROUP BY sentiment", session.get("user_id"))

    # Ratings
    rating_counts = db.execute("SELECT rating, COUNT(*) AS count FROM feedback WHERE user_id = ? GROUP BY rating", session.get("user_id"))

    # Tag Distribution
    tag_counts = db.execute("SELECT tag, COUNT(*) AS count FROM feedback WHERE user_id = ? GROUP BY tag", session.get("user_id"))

    # Recent Feedback
    total_rows = db.execute("SELECT COUNT(*) AS count FROM feedback WHERE user_id = ?", session.get("user_id"))[0]["count"]
    recent_feedback = db.execute("SELECT * FROM feedback WHERE user_id = ? ORDER BY created_at, id DESC LIMIT 5", session.get("user_id"))

    return render_template("index.html", total_rows=total_rows, recent_feedback=recent_feedback, rating_counts=rating_counts, summary=summary, sentiment_counts=sentiment_counts, topic_counts=topic_counts, tag_counts=tag_counts)


@app.route("/insights", methods=["GET", "POST"])
@login_required
def insights():
    if request.method == "POST":
        feedback = db.execute("SELECT * FROM feedback WHERE user_id = ?", session.get("user_id"))

        generate_ai_insights(feedback)

        return redirect("/insights")

    else:
        llm_output = db.execute("SELECT * FROM insights WHERE user_id = ? ORDER BY generated_at DESC LIMIT 1", session.get("user_id"))

        if not llm_output:
            return render_template("insights.html")

        issues = json.loads(llm_output[0]["issues"])
        recommendations = json.loads(llm_output[0]["recommendations"])

        return render_template("insights.html", llm_output=llm_output, issues=issues, recommendations=recommendations)


@app.route("/feedback")
@login_required
def feedback():
    """Display raw feedback"""
    ratings = db.execute("SELECT DISTINCT rating FROM feedback WHERE user_id = ? ORDER BY rating", session.get("user_id"))
    sentiments = db.execute("SELECT DISTINCT sentiment FROM feedback WHERE user_id = ?", session.get("user_id"))
    tags = db.execute("SELECT DISTINCT tag FROM feedback WHERE user_id = ?", session.get("user_id"))
    topics = db.execute("SELECT DISTINCT topic FROM feedback WHERE user_id = ?", session.get("user_id"))

    per_page = 10

    page = request.args.get("page", 1, type=int)
    rating = request.args.get("rating", type=int)
    sentiment = request.args.get("sentiment", type=str)
    topic = request.args.get("topic", type=str)
    tag = request.args.get("tag", type=str)

    offset = (page - 1) * per_page

    feedback_query = "SELECT * FROM feedback WHERE user_id = ?"
    total_rows_query = "SELECT COUNT(*) AS count FROM feedback WHERE user_id = ?"
    params = [session.get("user_id")]

    if rating:
        feedback_query += " AND rating = ?"
        total_rows_query += " AND rating = ?"
        params.append(rating)

    if sentiment:
        feedback_query += " AND sentiment = ?"
        total_rows_query += " AND sentiment = ?"
        params.append(sentiment)

    if topic:
        feedback_query += " AND topic = ?"
        total_rows_query += " AND topic = ?"
        params.append(topic)

    if tag:
        feedback_query += " AND tag = ?"
        total_rows_query += " AND tag = ?"
        params.append(tag)

    feedback_query += " ORDER BY id LIMIT ? OFFSET ?"

    feedback = db.execute(feedback_query, *params, per_page, offset)
    total_rows = db.execute(total_rows_query, *params)[0]["count"]
    total_pages = (total_rows + per_page - 1) // per_page

    start_page = ((page - 1) // 5) * 5 + 1
    end_page = min(start_page + 4, total_pages)

    return render_template("feedback.html", ratings=ratings, sentiments=sentiments, tags=tags, topics=topics,
                           feedback=feedback, page=page, rating=rating, sentiment=sentiment, topic=topic, tag=tag,
                           total_pages=total_pages, start_page=start_page, end_page=end_page)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Ensure username was submitted
        if not request.form.get("username"):
            error = "Please provide username"
            return render_template("login.html", error=error)

        # Ensure password was submitted
        elif not request.form.get("password"):
            error = "Please enter password"
            return render_template("login.html", error=error)

        # Query database for username
        rows = db.execute(
            "SELECT * FROM users WHERE username = ?", request.form.get("username")
        )

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(
            rows[0]["hash"], request.form.get("password")
        ):
            error = "Invalid username and/or password"
            return render_template("login.html", error=error)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        if not username:
            error = "Please provide username"
            return render_template("register.html", error=error)

        elif not password:
            error = "Please provide password"
            return render_template("register.html", error=error)

        elif not confirmation:
            error = "Please re-enter password"
            return render_template("register.html", error=error)

        elif password != confirmation:
            error = "Passwords do not match"
            return render_template("register.html", error=error)

        hash = generate_password_hash(password)

        try:
            db.execute("INSERT INTO users (username, hash) VALUES (?, ?)", username, hash)
        except ValueError:
            error = "Username already exists"
            return render_template("register.html", error=error)

        return redirect("/login")

    else:
        return render_template("register.html")


@app.route("/sources", methods=["GET", "POST"])
@login_required
def sources():
    """Upload feedback"""
    if request.method == "POST":
        file = request.files.get("feedback_raw")

        if not file:
            flash("No file uploaded", "danger")
            return redirect("/sources")
        if not file.filename.endswith(".csv"):
            flash("Only CSV files allowed", "danger")
            return redirect("/sources")

        # Implement CSV parsing
        decoded_file = io.StringIO(file.stream.read().decode("utf-8", errors="replace"))
        rows = list(csv.DictReader(decoded_file))

        # Validate uploaded data
        inserted_count = 0
        error_count = 0
        errors = []
        valid_rows = []
        for i, row in enumerate(rows):
            feedback = row.get("feedback_text")
            if not feedback:
                error_count += 1
                errors.append({"row": i + 1, "data": dict(row), "error": "Missing feedback"})
                continue
            preprocessed_feedback = preprocess_text(feedback)
            valid_rows.append((i, row, preprocessed_feedback))

        # Generate topics
        model, vectorizer = extract_topics([processed for _, _, processed in valid_rows])
        topics = get_topic_keywords(model, vectorizer)
        cluster_ids = model.labels_

        # Insert data into feedback table
        for (original_index, row, _), cluster_id in zip(valid_rows, cluster_ids):
            try:
                source = file.filename
                feedback_text = row["feedback_text"].strip()
                rating = row.get("rating")
                vote = row.get("vote")
                sentiment = analyze_sentiment(feedback_text)
                topic = ", ".join(topics[cluster_id])
                tag = assign_tag(feedback_text, rating)
                user_id = session.get("user_id")

                db.execute("INSERT INTO feedback (source, feedback_text, rating, feature_vote, sentiment, topic, tag, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", source, feedback_text, rating, vote, sentiment, topic, tag, user_id)

                inserted_count += 1

            except Exception as e:
                error_count += 1
                errors.append({"row": original_index + 1, "data": dict(row), "error": str(e)})

        flash(f"Uploaded {inserted_count} records successfully. View in Feedback tab.", "success")
        return render_template("sources.html", errors=errors, error_count=error_count)

    else:
        return render_template("sources.html")
