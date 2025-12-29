# WELFake Fake News Detection Project

## Project Overview
The goal of this project is to detect fake news using machine learning. The project uses the WELFake dataset, which contains labeled news articles (Real or Fake). This notebook demonstrates a full end-to-end ML workflow from data handling to model evaluation and hyperparameter tuning.

---

## 1. Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
```

**Explanation:**
- `pandas`, `numpy`: Data handling and numerical operations
- `matplotlib`, `seaborn`: Data visualization
- `scikit-learn`: ML algorithms, train/test split, vectorization, evaluation metrics
- `joblib`: Save/load models

---

## 2. Load and Explore Dataset
```python
df = pd.read_csv('welfake_cleaned.csv')
df.head()
df.info()
df['label'].value_counts()
```
- Check data shape, missing values, class distribution

---

## 3. Data Cleaning
- Ensure no missing values or duplicates
- Reset index
```python
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
```

---

## 4. Feature Selection and Split
```python
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- `X`: news text
- `y`: labels (0=Real, 1=Fake)
- Split data 80/20 for train/test

---

## 5. TF-IDF Vectorization
```python
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```
- Converts text to numerical features for ML models

---

## 6. Model Training
### 6.1 Logistic Regression
```python
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf, y_train)
y_pred_log = log_model.predict(X_test_tfidf)
print('Accuracy:', accuracy_score(y_test, y_pred_log))
```

### 6.2 Random Forest
```python
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
print('Accuracy:', accuracy_score(y_test, y_pred_rf))
```

### 6.3 Naive Bayes
```python
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
print('Accuracy:', accuracy_score(y_test, y_pred_nb))
```

---

## 7. Model Evaluation
- Use accuracy, classification report, confusion matrix
```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred_log))
cm = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
```
- Compare all three models side by side

---

## 8. Hyperparameter Tuning (Logistic Regression)
```python
param_grid = {'C':[0.01, 0.1, 1, 10], 'solver':['liblinear', 'lbfgs']}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_tfidf, y_train)
best_model = grid.best_estimator_
print('Best params:', grid.best_params_)
print('Test Accuracy:', accuracy_score(y_test, best_model.predict(X_test_tfidf)))
```
- Improves model performance and generalization

---

## 9. Test on Unseen News
```python
new_news = [
    'Government launches new environmental policy to reduce pollution.',
    'Aliens have landed in New York city, shocking everyone!'
]
new_news_tfidf = tfidf.transform(new_news)
predictions = best_model.predict(new_news_tfidf)
print('Predictions (0=Real, 1=Fake):', predictions)
```
- Demonstrates real-world application

---

## 10. Save Models and Vectorizer
```python
joblib.dump(log_model, 'welfake_log_model.pkl')
joblib.dump(rf_model, 'welfake_rf_model.pkl')
joblib.dump(nb_model, 'welfake_nb_model.pkl')
joblib.dump(tfidf, 'welfake_tfidf.pkl')
joblib.dump(best_model, 'welfake_log_best_model.pkl')
```
- Save trained models for future use

---

## ✅ Conclusion
- Logistic Regression (tuned) achieved **95.85% accuracy** on test set
- Random Forest and Naive Bayes provide alternative approaches
- Full end-to-end ML workflow demonstrated: **data cleaning → feature extraction → model training → evaluation → hyperparameter tuning → predictions → saving models**
- 
### Running the Notebook
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `WELFake Fake News Detection.ipynb` and run all cells to reproduce the analysis.


### Flask API
The project includes a Flask API for serving predictions,  (`model.pkl`)


#### Running the Flask API Locally
1. Ensure dependencies are installed (`requirements.txt`).
2. Run the Flask application:
   ```bash
   python app.py

   3. The API will be available at `http://localhost:5000`.

### Docker Deployment
The project includes a Dockerfile for containerized deployment.

#### Building and Running the Docker Container
1. Build the Docker image:
   ```bash
   docker build -t hospital-readmission-api .
   ```
2. Run the container:
   ```bash
   docker run -p 5000:5000 hospital-readmission-api
   ```
3. The API will be accessible at `http://localhost:5000`.

#### Notes
- The Dockerfile uses Python 3.9-slim for a lightweight image.
- Ensure `model.pkl`, `app.py`, and `requirements.txt` are in the project directory.