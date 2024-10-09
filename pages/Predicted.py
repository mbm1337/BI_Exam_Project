import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Indlæs data fra den uploadede CSV-fil
df = pd.read_csv('data/ai_job_market_insights.csv')

# Håndter NaN og numeriske værdier
df['Salary_USD'] = pd.to_numeric(df['Salary_USD'], errors='coerce')

# Vi beholder de oprindelige tekstlige værdier til dropdowns
original_df = df.copy()

# LabelEncode for kategoriske kolonner for at træne modellen
le_dict = {}
categorical_columns = ['Job_Title', 'Industry', 'Company_Size', 'Location', 'AI_Adoption_Level', 'Required_Skills', 'Remote_Friendly', 'Job_Growth_Projection']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le  # Gem label encoders for hver kolonne til senere

# Definer funktioner og mål
X = df.drop('Automation_Risk', axis=1)
y = df['Automation_Risk']

# Håndter NaN-værdier
X.fillna(X.mean(), inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Træn en DecisionTreeClassifier-model
clf = DecisionTreeClassifier(random_state=7)
clf.fit(X_train, y_train)

# Funktion til at forudsige automatiseringsrisiko
def predict_automation_risk(job_title, industry, company_size, location, ai_adoption, skills, salary, remote_friendly, growth_projection):
    # Opret DataFrame med brugerinput
    input_data = pd.DataFrame({
        'Job_Title': [job_title],
        'Industry': [industry],
        'Company_Size': [company_size],
        'Location': [location],
        'AI_Adoption_Level': [ai_adoption],
        'Required_Skills': [skills],
        'Salary_USD': [salary],
        'Remote_Friendly': [remote_friendly],
        'Job_Growth_Projection': [growth_projection]
    })

    # Encode kategoriske kolonner som i træningsdatasættet
    for col in categorical_columns:
        input_data[col] = le_dict[col].transform(input_data[col].astype(str))

    # Lav forudsigelsen
    prediction = clf.predict(input_data)
    return prediction[0]

# Byg Streamlit-appen
def main():
    st.title("Job Automation Risk Prediction")

    # Indsaml brugerinput fra dropdowns, hvor tekstlige unikke værdier fra datasættet bruges
    job_title = st.selectbox('Job Title', original_df['Job_Title'].unique())
    industry = st.selectbox('Industry', original_df['Industry'].unique())
    company_size = st.selectbox('Company Size', original_df['Company_Size'].unique())
    location = st.selectbox('Location', original_df['Location'].unique())
    ai_adoption = st.selectbox('AI Adoption Level', original_df['AI_Adoption_Level'].unique())
    skills = st.selectbox('Required Skills', original_df['Required_Skills'].unique())
    salary = st.number_input('Salary (in USD)', value=int(original_df['Salary_USD'].mean()))
    remote_friendly = st.selectbox('Remote Friendly', original_df['Remote_Friendly'].unique())
    growth_projection = st.selectbox('Job Growth Projection', original_df['Job_Growth_Projection'].unique())

    # Predict when the user clicks the button
    if st.button('Predict Automation Risk'):
        risk_prediction = predict_automation_risk(job_title, industry, company_size, location, ai_adoption, skills, salary, remote_friendly, growth_projection)
        st.success(f"The predicted automation risk for the job is: {risk_prediction}")

    # Evaluering og model resultater
    if st.checkbox('Show model evaluation'):
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy: {accuracy}')
        st.write('Classification Report:')
        st.text(classification_report(y_test, y_pred))

    # Vis feature importance
    if st.checkbox('Show feature importance'):
        importances = clf.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
        st.write(feature_importance_df)

# Streamlit app kører her
if __name__ == '__main__':
    main()
