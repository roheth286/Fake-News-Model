import numpy as np
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Prevent linter errors in IDE when analyzed in isolation
if 'train_df' not in globals():
    import pandas as pd
    train_df = pd.DataFrame()
if 'test_df' not in globals():
    import pandas as pd
    test_df = pd.DataFrame()

train_df.dropna(inplace=True)

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text

train_df['full_text'] = train_df['title'].fillna('') + " " + train_df['author'].fillna('') + " " + train_df['text'].fillna('')
test_df['full_text'] = test_df['title'].fillna('') + " " + test_df['author'].fillna('') + " " + test_df['text'].fillna('')

train_df['cleaned_text'] = train_df['full_text'].apply(clean_text)
test_df['cleaned_text'] = test_df['full_text'].apply(clean_text)

print("Cleaned Training Data Head:")
print(train_df[['cleaned_text', 'label']].head())
print("\nCleaned Testing Data Head:")
print(test_df[['cleaned_text']].head())
