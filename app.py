from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example resume and job description text
resume = """Experienced Python developer with strong knowledge in data analysis, pandas, and web development using Flask. Skilled in creating web applications and managing databases."""
job_desc = """Looking for a Python developer skilled in Flask, APIs, and data processing using pandas. Experience with databases and web app development is a plus."""

# Convert text to numbers using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume, job_desc])

# Calculate cosine similarity between the resume and job description
similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# Output the similarity percentage
print(f"Resume Match Score: {round(similarity * 100, 2)}%")

