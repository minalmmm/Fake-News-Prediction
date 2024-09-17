import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# Function to preprocess the text
def preprocess_text(text):
    # Add your text preprocessing here (e.g., lowercasing, removing punctuation)
    return text.lower()

# Function to build and train the model
def build_and_train_model(data):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))),
        ('clf', MultinomialNB())
    ])
    
    X_train = data['text']
    y_train = data['label']
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    return pipeline

# Define the main function
def main():
    st.title("News Classification App")

    # Option to upload a CSV file or enter news text directly
    upload_option = st.radio("Choose an option:", ("Upload CSV File", "Enter News Text"))

    if upload_option == "Upload CSV File":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                news_data = pd.read_csv(uploaded_file)
                st.write("CSV File Uploaded Successfully")
                st.write(news_data.head())

                # Ensure the required columns are present
                if 'text' not in news_data.columns or 'label' not in news_data.columns:
                    st.error("CSV file must contain 'text' and 'label' columns.")
                    return
                
                # Build and train the model
                model = build_and_train_model(news_data)
                
                if st.button("Predict"):
                    news_data['predictions'] = model.predict(news_data['text'])
                    st.write(news_data[['text', 'predictions']])
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif upload_option == "Enter News Text":
        news_text = st.text_area("Enter the news text")
        if st.button("Predict"):
            if news_text:
                # Create a dummy model for prediction
                model = build_and_train_model(pd.DataFrame({'text': [news_text], 'label': ['']*1}))
                processed_text = preprocess_text(news_text)
                text_vector = model.named_steps['tfidf'].transform([processed_text])
                prediction = model.named_steps['clf'].predict(text_vector)
                st.write(f"Predicted Label: {prediction[0]}")
            else:
                st.error("Please enter some text")

if __name__ == "__main__":
    main()










