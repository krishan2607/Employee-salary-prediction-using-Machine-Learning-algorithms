# 💼 Employee Salary Prediction using Machine Learning

This project aims to predict whether an employee earns more than ₹50K annually using various demographic and work-related features. It uses machine learning classification techniques and is deployed as an interactive web application using **Streamlit**.

## 📌 Problem Statement

Organizations often require tools for compensation analysis and HR decision-making. This project addresses the need to classify employees into two salary groups:
- `<=50K`
- `>50K`

The model is trained on real-world data (`adult.csv`) and uses attributes such as:
- Age
- Work class
- Education
- Occupation
- Gender
- Capital Gain / Loss
- Hours per week
- Native country
- Years of experience

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – data processing
  - `scikit-learn` – model building, evaluation
  - `joblib` – saving/loading models and encoders
  - `streamlit` – interactive web UI
- **Tools**: Jupyter Notebook, VS Code
- **Deployment**: Streamlit (locally or cloud)

---

## 🧠 Model & Approach

### 1. **Data Preprocessing**
- Missing values replaced or dropped
- Label Encoding for categorical variables
- Feature Scaling using StandardScaler
- Splitting into training/testing sets (80/20)

### 2. **Model Building**
- Trained multiple classifiers:
  - Random Forest Classifier (final choice)
  - Logistic Regression
  - SVM
- Evaluated using:
  - Accuracy
  - F1-Score
  - Precision / Recall

### 3. **Model Saving**
- Saved model (`best_model.pkl`), scaler, encoders, and expected column layout using `joblib`

---

## 🌐 Web App (Streamlit)

The `app.py` Streamlit application allows users to:
- Enter employee details manually
- Preview the encoded and scaled input
- Predict the salary class in real-time

### Features
- Slider & dropdown inputs
- Model loaded with `joblib`
- Interactive layout with prediction output

---

## 🖥️ How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/employee-salary-prediction.git
   cd employee-salary-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

```
📦employee-salary-prediction
├── app.py                   # Streamlit web app
├── adult.csv                # Dataset
├── ModelTraining.ipynb      # Jupyter notebook for training & evaluation
├── best_model.pkl           # Trained model
├── adult_scaler.pkl         # Scaler used during preprocessing
├── adult_encoders.pkl       # Label encoders for categorical columns
├── adult_columns.pkl        # Expected columns after preprocessing
├── Project.pptx             # Project presentation
└── README.md                # Project overview and documentation
```

---

## 📊 Results

The final Random Forest model achieved:
- **Accuracy**: ~86%
- **F1-Score**: High, depending on class balance

---

## 📌 Future Enhancements

- Add more features like marital status, relationship, etc.
- Use deep learning or ensemble methods to improve accuracy
- Deploy to cloud platforms like Heroku or Hugging Face Spaces
- Integrate database for logging inputs/predictions

---

## 🔗 References

- [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Scikit-learn documentation
- Streamlit documentation

---

## 🙋‍♂️ Author

**Krishan Jakhar**  
Bikaner Technical University  
Department of Electronics and Communication Engineering  
🔗 [LinkedIn](https://www.linkedin.com/in/krishan-jakhar-a58678292/)
