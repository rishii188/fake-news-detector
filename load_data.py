import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import joblib

# Load each dataset
fake_df = pd.read_csv('Fake.csv')[['title', 'text', 'subject', 'date']]
true_df = pd.read_csv('True.csv')[['title', 'text', 'subject', 'date']]

# Add labels
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine into one dataframe
df = pd.concat([fake_df, true_df], ignore_index=True)[['title', 'text', 'subject', 'date', 'label']]

# Shuffle the rows
df = df.sample(frac=1).reset_index(drop=True)

# Save the combined dataframe to a new CSV file
df.to_csv('combined_news_data.csv', index=False)

# Display the first few rows of the dataframe
print('\t\t\t -- Combined DataFrame: -- ')
print(df.head())

# Display the structure of the dataframe
print('\n\t\t\t -- DataFrame Info: -- ')
print(df.info())

# Display the distribution of labels
print('\n\t\t\t -- Label Distribution: -- ')
print(df['label'].value_counts())

# Check for missing values
print('\n\t\t\t -- Missing Values: -- ')
print(df.isnull().sum())

# Check summary statistics
print('\n\t\t\t -- Summary Statistics: -- ')
print(df.describe(include='all'))

# Title length analysis (fake news)
fake_df['title_length'] = fake_df['title'].apply(len)
sns.histplot(fake_df['title_length'], bins=30, kde=True)
plt.title('Distribution of Title Lengths (Fake News)')
plt.xlabel('Title Length')
plt.ylabel('Frequency')
plt.show()

# Text length analysis (fake news)
fake_df['text_length'] = fake_df['text'].apply(len)
sns.histplot(fake_df['text_length'], bins=30, kde=True)
plt.title('Distribution of Text Lengths (Fake News)')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Title length analysis (real news)
true_df['title_length'] = true_df['title'].apply(len)
sns.histplot(true_df['title_length'], bins=30, kde=True)
plt.title('Distribution of Title Lengths (Real News)')
plt.xlabel('Title Length')
plt.ylabel('Frequency')
plt.show()

# Text length analysis (real news)
true_df['text_length'] = true_df['text'].apply(len)
sns.histplot(true_df['text_length'], bins=30, kde=True)
plt.title('Distribution of Text Lengths (Real News)')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Load combined dataset
df = pd.read_csv('combined_news_data.csv')

# Combine title and text
df['content'] = df['title'] + " " + df['text']

# Encode labels (FAKE=0, REAL=1)
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

print(df[['content', 'label']].head())

x = df['content']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Initialise TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit on train data and transform both train and test data
X_train_tfidf = tfidf.fit_transform(x_train)
X_test_tfidf = tfidf.transform(x_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_tfidf)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save
plt.show()

# Save model + vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Value': [accuracy, precision, recall, f1]
})
metrics_df.to_csv('model_metrics.csv', index=False)

# Store results for comparison
results = []

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)
y_pred_lr = log_reg.predict(X_test_tfidf)
results.append({
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-score': f1_score(y_test, y_pred_lr)
})

# Linear SVM
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
results.append({
    'Model': 'Linear SVM',
    'Accuracy': accuracy_score(y_test, y_pred_svm),
    'Precision': precision_score(y_test, y_pred_svm),
    'Recall': recall_score(y_test, y_pred_svm),
    'F1-score': f1_score(y_test, y_pred_svm)
})

# XGBoost
xgb_model = XGBClassifier(eval_metric='logloss', tree_method='hist', random_state=42)
xgb_model.fit(X_train_tfidf, y_train)
y_pred_xgb = xgb_model.predict(X_test_tfidf)
results.append({
    'Model': 'XGBoost',
    'Accuracy': accuracy_score(y_test, y_pred_xgb),
    'Precision': precision_score(y_test, y_pred_xgb),
    'Recall': recall_score(y_test, y_pred_xgb),
    'F1-score': f1_score(y_test, y_pred_xgb)
})

# Create DataFrame for comparison
print('\n\t\t\t -- Model Comparison: -- ')
comparison_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
print(comparison_df)

# Save comparison
comparison_df.to_csv('model_comparison.csv', index=False)