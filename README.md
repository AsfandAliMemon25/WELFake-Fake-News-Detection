# Predicting on WELFake News Detection

## Project Overview
This project demonstrates a complete end-to-end machine learning workflow for detecting fake news
using the WELFake dataset. The workflow includes data cleaning, feature extraction, model training,
evaluation, hyperparameter tuning, testing on new data, and saving models for deployment.
2. **Develop an accurate predictive model** using a limited set of features.


### Data Preprocessing
- **Missing Values**: Replaced '?' with 'Unknown' or 'Missing' for categorical variables (e.g., race, diagnoses). Dropped columns with excessive missing data (e.g., weight, payer_code).

## Methodology
The project follows a structured approach to analyze the dataset, identify predictors, and build predictive models.

### Steps
1. **Data Exploration**:
   - Analyzed dataset for missing values, distributions, and imbalances.
   

2. **Feature Engineering**:
   - 
   - Applied log transformations to skewed features.

3. **Modeling**:
   - **Feature Selection**: Used Recursive Feature Elimination (RFE) with RandomForest to select the top 30 features.
   - **Data Balancing**: Applied ADASYN to address class imbalance (88% non-readmitted vs. 12% readmitted).
   - **Models Tested**:
     - Decision Tree
     - Logistic Regression
     - Naive Bias 
   - **Hyperparameter Tuning**: Used GridSearchCV with cross-validation to optimize model performance.
   - **Evaluation Metrics**: Focused on recall (to capture readmissions), precision, F1-score, ROC-AUC, and PR-AUC.

4. **Final Model**:
   - Selected Naive Bias with tuned hyperparameters for its balance of performance and interpretability.
   - Adjusted prediction threshold to prioritize recall ≥ 0.70 while maximizing precision.

5. **Deployment**:
   - Developed a Flask API to serve predictions.
   - Containerized the application using Docker for scalable deployment.

6. **Visualization**:
   - Exported predictions to CSV for visualization in Tableau, enabling interactive dashboards.

### Models and Hyperparameters
| **Model**       | **Best Hyperparameters**                                                                 | **Scoring Metric** |
|-----------------|-----------------------------------------------------------------------------------------|---------------------|
| Decision Tree   | `criterion='gini', max_depth=7, min_samples_leaf=2, min_samples_split=5`                | F1-score            |
| Logistic Regression         | `learning_rate=0.05, max_depth=5, n_estimators=100, scale_pos_weight=3, subsample=0.8`  | F1-score            |
| Logistic Regression       | `learning_rate=0.05, max_depth=5, n_estimators=100, num_leaves=20`                     | Recall              |



## Installation and Usage
### Prerequisites
- Python 3.9+
- Docker (for containerized deployment)
- Tableau (for visualization)
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `flask`, `imblearn`, `seaborn`, `matplotlib`


2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ````.

### Running the Notebook
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `Prediction on WELfake News Detection.ipynb` and run all cells to reproduce the analysis.

### Flask API
The project includes a Flask API for serving predictions, using the trained XGBoost model (`model.pkl`).

#### Running the Flask API Locally
1. Ensure dependencies are installed (`requirements.txt`).
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. The API will be available at `http://localhost:5000`.

#### Making Predictions
- **Endpoint**: `/predict` (POST)
- **Input**: JSON array of feature dictionaries (see `test.json` for an example).
- **Output**: JSON with predictions (0 or 1) and probabilities for readmission.
- **Example Request**:
  ```bash
  curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @test.json
  ```
- **Example Response**:
  ```json
  {
    "predictions": [0],
    "probabilities": [0.123456]
  }
  ```
- Logs are saved to `app.log` for debugging.

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

.
