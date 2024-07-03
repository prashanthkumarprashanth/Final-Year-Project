import streamlit as st
import sqlite3
from passlib.hash import pbkdf2_sha256
from streamlit_option_menu import option_menu
import requests
import webbrowser
# Import Pandas for data manipulation using dataframes
import pandas as pd  
# Import TF-IDF vectorizer for converting text data into a matrix of TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer  
# Import BeautifulSoup for web scraping
from bs4 import BeautifulSoup  
# Import Requests for making HTTP requests
# Import Matplotlib for data visualization
import matplotlib.pyplot as plt  
# Import Seaborn for statistical data visualization
import seaborn as sns  
# Import WordCloud for generating word clouds
from wordcloud import WordCloud  
# Import Plotly Express for interactive visualizations
import plotly.express as px  
import requests
import random
from bs4 import BeautifulSoup  
from sklearn.metrics.pairwise import linear_kernel
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Function to create database connection
# Read City dataset from CSV
df = pd.read_csv("City.csv")

# Combine relevant features into a single text column for TF-IDF vectorization
df['combined_features'] = df['Best Time'] + ' ' + df['City_desc']

# Use TF-IDF vectorizer to convert the text data into a matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
# Function to search and display images based on a query using Google Images
def search_and_display_images(query, num_images=20):
    try:
        # Initialize an empty list for image URLs
        k=[]  
        # Initialize an index for iterating through the list of images
        idx=0  
        # Construct Google Images search URL
        url = f"https://www.google.com/search?q={query}&tbm=isch"  
         # Make an HTTP request to the URL
        response = requests.get(url) 
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")  
        # Initialize an empty list for storing image URLs
        images = []  
        # Iterate through image tags in the HTML content
        for img in soup.find_all("img"):  
             # Limit the number of images to the specified amount
            if len(images) == num_images: 
                break
            # Get the image source URL
            src = img.get("src")  
            # Check if the source URL is valid
            if src.startswith("http") and not src.endswith("gif"):  
                # Add the image URL to the list
                images.append(src)  
        # Iterate through the list of image URLs
        for image in images:  
            # Add each image URL to the list 'k'
            k.append(image)  
        # Reset the index for iterating through the list of image URLs
        idx = 0  
        # Iterate through the list of image URLs
        while idx < len(k):
            # Iterate through the columns in a 4-column layout 
            for _ in range(len(k)): 
                # Create 4 columns for displaying images 
                cols = st.columns(4)  
                # Display the first image in the first column
                cols[0].image(k[idx], width=150)  
                idx += 1 
                # Move to the next image in the list
                cols[1].image(k[idx], width=150)
                # Display the second image in the second column
                idx += 1  
                # Move to the next image in the list
                cols[2].image(k[idx], width=150)  
                # Display the third image in the third column
                idx += 1  
                # Move to the next image in the list
                cols[3].image(k[idx], width= 150)  
                # Display the fourth image in the fourth column
                idx = idx + 1  
                # Move to the next image in the list
    except:
         # Handle exceptions gracefully if there is an error while displaying images
        pass  
   

# Function to get recommendations based on user input
def get_recommendations(user_best_time):
    # Combine user input into a similar text format
    user_input = user_best_time
    
    # Transform user input using the TF-IDF vectorizer
    user_tfidf = tfidf_vectorizer.transform([user_input])
    
    # Calculate dot product between user input and existing items
    similarity_scores = user_tfidf.dot(tfidf_matrix.T)
    
    # Get indices of items with positive similarity scores
    similar_indices = similarity_scores.indices
    
    # Get recommendations with description
    recommendations = df.loc[similar_indices, ['City', 'City_desc']]
    # returning the recommendations
    return recommendations
def create_connection():
    conn = sqlite3.connect('health_app.db')
    return conn
def create_user_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        );
    ''')
    conn.commit()

# Function to insert user data into user table
def insert_user(conn, username, password):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users (username, password) VALUES (?, ?);
    ''', (username, pbkdf2_sha256.hash(password)))
    conn.commit()

# Function to check if user exists in user table
def check_user(conn, username, password):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT password FROM users WHERE username = ?;
    ''', (username,))
    user = cursor.fetchone()
    if user:
        return pbkdf2_sha256.verify(password, user[0])
    return False

# UI for login page
def login_page(conn):
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            if check_user(conn, username, password):
                linkedin_url = "https://travel-g4hrr4h68ds5xh3q9jgzzx.streamlit.app/"
                st.success("Redirecting to Recommendations page...")
                webbrowser.open_new_tab(linkedin_url)
            else:
                st.error("Invalid username or password.")
# Function to retrieve daily updates by username
def retrieve_daily_updates(conn, username):
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM daily_updates WHERE username = ?;
    ''', (username,))
    return cursor.fetchall()
# UI for registration page
def register_page(conn):
    st.header("Register")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if new_username and new_password and new_password == confirm_password:
            insert_user(conn, new_username, new_password)
            st.success("Registered successfully! You can now login.")
        else:
            st.error("Please fill out all fields and ensure passwords match.")
# UI for Daily Updates History page# Main UI
st.set_page_config(page_title="Travel Application", page_icon=":hospital:")

st.write("<h1 style='color: green;'>TRAVEL RECOMMENDATION SYSTEM</h1>", unsafe_allow_html=True)

with st.sidebar:
    page = option_menu("DashBoard", ["Home",'Login','Register','Guest Login'], 
        icons=['house','unlock-fill','lock-fill','person'], menu_icon="cast", default_index=0,
        styles={
        "nav-link-selected": {"background-color": "green", "color": "white", "border-radius": "5px"},
})

conn = create_connection()
create_user_table(conn)
if page == "Home":
    st.image('1.jpg',use_column_width=True)
if page == "Login":
    login_page(conn)
elif page == "Register":
    register_page(conn)
elif page == "Guest Login":
    user_best_time = st.selectbox("Select the best time to visit", df['Best Time'].unique())
    recommendations = get_recommendations(user_best_time)

    # Display recommendations in a dropdown
    selected_recommendation = st.selectbox("Select a destination", recommendations['City'].tolist())
    genre = st.radio(
    "Select Option:",
    [":rainbow[Place]", ":rainbow[Near Places]", ":rainbow[Accommodations]"])

    if genre == ":rainbow[Place]":   

        # Display selected recommendation and its description
        st.subheader("Selected Recommendation:")
        # Display the selected recommendation
        selected_description = recommendations.loc[recommendations['City'] == selected_recommendation, 'City_desc'].iloc[0]
        # Display the description of the selected recommendation
        st.write(f"**{selected_recommendation}**: {selected_description}")
        # Display the selected recommendation
        search_and_display_images(selected_recommendation)
    if genre == ":rainbow[Near Places]":
        # Read Places dataset from CSV
        df1=pd.read_csv("Places.csv")
        # Display the nearest places
        st.subheader("Nearest Places:")
        # Display the nearest places
        res=df1.loc[df1['City'] == selected_recommendation]
        # Display the nearest places
        for index, row in res.iterrows():
            st.write( row['Place'])
            st.write("Distance: ",row['Distance'])
            st.write(row['Place_desc'])
            search_and_display_images(row['Place'],4)
    if genre == ":rainbow[Accommodations]":
        # Read Travel Cost dataset from CSV
        df2=pd.read_csv('travel cost.csv')
        # Display the accommodation cost
        st.subheader("Accommodation Cost:")
        # Display the accommodation cost
        res1=df2.loc[df2['City'] == selected_recommendation]
        # get the accommodation cost for the selected recommendation
        for index, row in res1.iterrows():
            st.write( row['Accomadation_Type'],":",row['Accomdation_Cost'])
            text= row['Accomadation_Type']+"in"+selected_recommendation
            st.write(row['source'])
            search_and_display_images(text,4)
    
