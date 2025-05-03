from flask import Flask, render_template,request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Route to show the resume matcher form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the form submission
@app.route('/match', methods=['POST'])
def match():
    # Get resume and job description from the form
    resume = request.form['resume']
    job_desc = request.form['job_desc']

    # Create TF-IDF Vectorizer and transform the input texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume, job_desc])
    
    # Calculate the cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    #convert the similarity into percentage
    similarity_percentage = round(similarity[0][0] * 100, 2)
    similarity_str = f"{similarity_percentage}%"

    # Return the similarity score to the user
    return render_template('index.html', similarity=similarity_str)

if __name__ == '__main__':
    app.run(debug=True)
