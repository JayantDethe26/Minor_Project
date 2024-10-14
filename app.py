from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, classification_report

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r'C:\Users\dethe\.vscode\VSCode\StudentPerformance\datset.csv')

# Preprocess the data
df.fillna({
    'Age': df['Age'].mean(),
    'Gender': df['Gender'].mode()[0],
    'Ethnicity': df['Ethnicity'].mode()[0],
    'ParentalEducation': df['ParentalEducation'].mode()[0],
    'StudyTimeWeekly': df['StudyTimeWeekly'].mean(),
    'Absences': df['Absences'].mean(),
    'Tutoring': df['Tutoring'].mode()[0],
    'ParentalSupport': df['ParentalSupport'].mode()[0],
    'Extracurricular': df['Extracurricular'].mode()[0],
    'Sports': df['Sports'].mode()[0],
    'CGPA': df['CGPA'].mean(),
    'Grade': df['Grade'].mode()[0]
}, inplace=True)

# Feature engineering
df['StudyAbsenceRatio'] = df['StudyTimeWeekly'] / (df['Absences'] + 1)  # Avoid division by zero
df['ParentalSupportFactor'] = df['ParentalSupport'] * 1.5  # Assuming a higher weight for parental support

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Ethnicity', 'ParentalEducation',
                       'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Grade']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare features and target
feature_columns = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation',
                   'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport',
                   'Extracurricular', 'Sports', 'StudyAbsenceRatio', 'ParentalSupportFactor']
X = df[feature_columns]
y_cgpa = df['CGPA']
y_grade = df['Grade']

# Split the data into training and testing sets
X_train_cgpa, X_test_cgpa, y_train_cgpa, y_test_cgpa = train_test_split(X, y_cgpa, test_size=0.3, random_state=42)
X_train_grade, X_test_grade, y_train_grade, y_test_grade = train_test_split(X, y_grade, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_cgpa_scaled = scaler.fit_transform(X_train_cgpa)
X_test_cgpa_scaled = scaler.transform(X_test_cgpa)

# Hyperparameter tuning for Gradient Boosting Regressor (CGPA Prediction)
regressor = GradientBoostingRegressor(random_state=42)
param_grid_cgpa = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
grid_search_cgpa = GridSearchCV(estimator=regressor, param_grid=param_grid_cgpa, cv=5, verbose=1, n_jobs=-1)
grid_search_cgpa.fit(X_train_cgpa_scaled, y_train_cgpa)
best_regressor = grid_search_cgpa.best_estimator_

# Hyperparameter tuning for Gradient Boosting Classifier (Grade Prediction)
clf = GradientBoostingClassifier(random_state=42)
param_grid_grade = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
grid_search_grade = GridSearchCV(estimator=clf, param_grid=param_grid_grade, cv=5, verbose=1, n_jobs=-1)
grid_search_grade.fit(X_train_grade, y_train_grade)
best_clf = grid_search_grade.best_estimator_

# Evaluate the best models
y_pred_cgpa = best_regressor.predict(X_test_cgpa_scaled)
mse_cgpa = mean_squared_error(y_test_cgpa, y_pred_cgpa)
print("CGPA Prediction Mean Squared Error:", mse_cgpa)

y_pred_grade = best_clf.predict(X_test_grade)
print(classification_report(y_test_grade, y_pred_grade))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collecting input data from form
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        ethnicity = int(request.form['ethnicity'])
        parental_education = int(request.form['parental_education'])
        study_time_weekly = float(request.form['study_time_weekly'])
        absences = int(request.form['absences'])
        tutoring = int(request.form['tutoring'])
        parental_support = int(request.form['parental_support'])
        extracurricular = int(request.form['extracurricular'])
        sports = int(request.form['sports'])

        # Calculate engineered features
        study_absence_ratio = study_time_weekly / (absences + 1)  # Avoid division by zero
        parental_support_factor = parental_support * 1.5  # Assuming a higher weight for parental support

        # Create new data for prediction including engineered features
        new_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Ethnicity': ethnicity,
            'ParentalEducation': parental_education,
            'StudyTimeWeekly': study_time_weekly,
            'Absences': absences,
            'Tutoring': tutoring,
            'ParentalSupport': parental_support,
            'Extracurricular': extracurricular,
            'Sports': sports,
            'StudyAbsenceRatio': study_absence_ratio,
            'ParentalSupportFactor': parental_support_factor
        }])

        # Scale the new data
        new_data_scaled = scaler.transform(new_data)

        # Make predictions
        predicted_cgpa = best_regressor.predict(new_data_scaled)[0]
        predicted_grade = best_clf.predict(new_data)[0]
        grade_label = label_encoders['Grade'].inverse_transform([predicted_grade])[0]

        return render_template('index.html', predicted_cgpa=predicted_cgpa, predicted_grade=grade_label)

    return render_template('index.html', predicted_cgpa=None, predicted_grade=None)

if __name__ == '__main__':
    app.run(debug=True)