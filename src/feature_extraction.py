from sklearn.feature_extraction.text import TfidfVectorizer

# Prevent linter errors in IDE when analyzed in isolation
if 'train_df' not in globals():
    import pandas as pd
    train_df = pd.DataFrame()
if 'test_df' not in globals():
    import pandas as pd
    test_df = pd.DataFrame()

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['cleaned_text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['cleaned_text'])
y_train = train_df['label'].values

print(f"Train Data Shape: {X_train_tfidf.shape}")
print(f"Test Data Shape: {X_test_tfidf.shape}")
