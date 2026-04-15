import re
import string
import nltk
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    
    # remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove extra spaces
    text = text.strip()

    return text