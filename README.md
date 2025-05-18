import os
from flask import Flask, request, render_template
from dotenv import load_dotenv
import openai
import requests
from transformers import pipeline
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

app = Flask(__name__)

# Initialize emotion classifier
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=False)

def analyze_emotion(user_input):
    result = emotion_classifier(user_input)
    return result[0]['label']

def get_movie_recommendations(emotion):
    prompt = f"Suggest three movies that match the {emotion} mood."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a movie recommendation assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].split('\n')

def fetch_movie_details(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            movie = data['results'][0]
            return {
                'title': movie.get('title'),
                'overview': movie.get('overview'),
                'poster': f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None,
                'link': f"https://www.themoviedb.org/movie/{movie.get('id')}"
            }
    return {'title': movie_title, 'overview': 'No details found.', 'poster': None, 'link': '#'}

def send_email(to_email, emotion, movies):
    movie_list = ''.join([f"<li><a href='{movie['link']}'>{movie['title']}</a></li>" for movie in movies])
    message = Mail(
        from_email='your_email@example.com',
        to_emails=to_email,
        subject='Your Personalized Movie Recommendations',
        html_content=f"""
        <h2>Your detected mood: {emotion}</h2>
        <h3>Recommended Movies:</h3>
        <ul>{movie_list}</ul>
        """
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
    except Exception as e:
        print(e)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['mood']
        email = request.form['email']
        emotion = analyze_emotion(user_input)
        recommendations = get_movie_recommendations(emotion)
        movies = [fetch_movie_details(title) for title in recommendations if title]
        send_email(email, emotion, movies)
        return render_template('result.html', emotion=emotion, movies=movies)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
