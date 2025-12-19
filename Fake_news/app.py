import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -------------------------------
# Create folder to save plots
# -------------------------------
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# -------------------------------
# Load Dataset
# -------------------------------
print("\nLoading dataset...")
data = pd.read_csv("data/WELFake_Dataset.csv").dropna()

# -------------------------------
# Balanced Subset (8000 samples)
# -------------------------------
real_news = data[data['label'] == 0].sample(n=4000, random_state=42)
fake_news = data[data['label'] == 1].sample(n=4000, random_state=42)
data = pd.concat([real_news, fake_news]).sample(frac=1, random_state=42)
print("Balanced subset selected (4000 Real + 4000 Fake)")
print("Total samples:", len(data))

# -------------------------------
# Combine title and text
# -------------------------------
if 'title' in data.columns:
    data['content'] = data['title'] + " " + data['text']
else:
    data['content'] = data['text']

X = data['content']
y = data['label']

# -------------------------------
# Class Distribution Visualization
# -------------------------------
plt.figure(figsize=(6,4))
ax = sns.countplot(x=y, palette="pastel")
plt.title("Class Distribution (0 = Real, 1 = Fake)")
plt.xlabel("News Type")
plt.ylabel("Count")
total = len(y)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y_pos = p.get_height()
    ax.annotate(percentage, (x, y_pos), ha='center', va='bottom')
plt.savefig(os.path.join(plots_dir, "class_distribution.png"))
plt.close()

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
print("\nApplying TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(X)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# -------------------------------
# Logistic Regression Model
# -------------------------------
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)

# -------------------------------
# Multinomial Naive Bayes Model
# -------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# -------------------------------
# Hybrid Model Prediction
# -------------------------------
def hybrid_predict(X):
    log_pred = log_model.predict(X)
    nb_pred = nb_model.predict(X)
    # If both agree, take prediction; else take Logistic Regression
    final_pred = [l if l == n else l for l, n in zip(log_pred, nb_pred)]
    return np.array(final_pred)

# -------------------------------
# Model Evaluation
# -------------------------------
y_train_pred = hybrid_predict(X_train)
y_test_pred = hybrid_predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

def score_to_stars(score, max_stars=10):
    stars = int(round(score * max_stars))
    return "★" * stars + "☆" * (max_stars - stars)

metrics_table = [
    ["Training Accuracy", f"{train_acc:.4f}", score_to_stars(train_acc)],
    ["Testing Accuracy", f"{test_acc:.4f}", score_to_stars(test_acc)],
    ["Precision", f"{precision:.4f}", score_to_stars(precision)],
    ["Recall", f"{recall:.4f}", score_to_stars(recall)],
    ["F1 Score", f"{f1:.4f}", score_to_stars(f1)]
]

print("\nMODEL EVALUATION METRICS (HYBRID MODEL):\n")
print(tabulate(metrics_table, headers=["Metric", "Score", "Stars"], tablefmt="fancy_grid"))

# -------------------------------
# Accuracy Comparison Bar Chart
# -------------------------------
plt.figure(figsize=(6,4))
plt.bar(["Training Accuracy", "Testing Accuracy"], [train_acc, test_acc], color=['skyblue','orange'])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy (Hybrid Model)")
plt.savefig(os.path.join(plots_dir, "accuracy_comparison.png"))
plt.close()

# -------------------------------
# Confusion Matrix Heatmap
# -------------------------------
cm = confusion_matrix(y_test, y_test_pred)
cm_percent = cm / cm.sum(axis=1)[:, None]

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
plt.title("Confusion Matrix (Count)")
plt.savefig(os.path.join(plots_dir, "confusion_matrix_count.png"))
plt.close()

plt.figure(figsize=(6,4))
sns.heatmap(cm_percent, annot=True, fmt=".2%", cmap="Blues", xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
plt.title("Confusion Matrix (Percent)")
plt.savefig(os.path.join(plots_dir, "confusion_matrix_percent.png"))
plt.close()

# -------------------------------
# Top TF-IDF Features
# -------------------------------
feature_names = vectorizer.get_feature_names_out()
coefficients = log_model.coef_[0]
top_n = 15

top_fake_idx = np.argsort(coefficients)[-top_n:]
top_real_idx = np.argsort(coefficients)[:top_n]

# Bar plots for top words
plt.figure(figsize=(10,6))
plt.barh([feature_names[i] for i in top_fake_idx], [coefficients[i] for i in top_fake_idx], color='salmon')
plt.title("Top Words Indicating Fake News")
plt.xlabel("Coefficient Weight")
plt.gca().invert_yaxis()
plt.savefig(os.path.join(plots_dir, "top_words_fake.png"))
plt.close()

plt.figure(figsize=(10,6))
plt.barh([feature_names[i] for i in top_real_idx], [coefficients[i] for i in top_real_idx], color='lightgreen')
plt.title("Top Words Indicating Real News")
plt.xlabel("Coefficient Weight")
plt.gca().invert_yaxis()
plt.savefig(os.path.join(plots_dir, "top_words_real.png"))
plt.close()

# -------------------------------
# Prediction Distribution Pie Chart
# -------------------------------
pred_counts = pd.Series(y_test_pred).value_counts()
plt.figure(figsize=(6,6))
plt.pie(pred_counts, labels=["Real News","Fake News"], autopct='%1.1f%%', colors=['lightgreen','salmon'])
plt.title("Prediction Distribution on Test Set")
plt.savefig(os.path.join(plots_dir, "prediction_distribution.png"))
plt.close()

# -------------------------------
# Save Hybrid Models & Vectorizer
# -------------------------------
pickle.dump(log_model, open("logistic_model.pkl", "wb"))
pickle.dump(nb_model, open("naive_bayes_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
print("\nHybrid models and vectorizer saved successfully")
