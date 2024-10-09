# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data/ai_job_market_insights.csv')

# Data Info Section
st.title("AI Job Market Insights Analysis")
st.write("### Dataset Overview")
st.write("""
Dette datasæt indeholder indsigt i AI-relaterede job, herunder jobtitler, industrier, lønninger, AI-adoptionsniveauer og meget mere. 
Vi vil analysere disse data gennem visualiseringer for at få en dybere forståelse af jobmarkedet.
""")
st.write(df.head())


# Descriptive Stats
st.write("### Descriptive Statistics")
st.write("""
De følgende deskriptive statistikker viser nogle grundlæggende statistiske oplysninger om de numeriske kolonner i datasættet, 
såsom middelværdi, standardafvigelse, minimum- og maksimumværdier.
""")
st.write(df.describe())

# Visualize the distribution of Salary
st.write("### Distribution of Salary")
st.write("""
Diagrammet nedenfor viser fordelingen af lønninger i USD for job der mugligvis kan erstats af Ai. Histogrammet viser antallet af job i forskellige lønintervaller, 
mens kurven over (KDE-kurven) hjælper med at forstå den glatte fordeling af lønninger.
""")
plt.figure(figsize=(10, 6))
sns.histplot(df['Salary_USD'], kde=True, bins=30)
plt.title('Distribution of Salary')
plt.xlabel('Salary')
plt.ylabel('Frequency')
st.pyplot(plt)

# Visualize the count of Job Titles
st.write("### Count of Job Titles")
st.write("""
Dette diagram viser antallet af jobtitler i datasættet. Det hjælper os med at se, hvilke jobtitler der er mest almindelige, og giver et overblik over mangfoldigheden af roller på markedet.
""")
plt.figure(figsize=(12, 8))
sns.countplot(y='Job_Title', data=df, order=df['Job_Title'].value_counts().index)
plt.title('Count of Job Titles')
plt.xlabel('Count')
plt.ylabel('Job Title')
st.pyplot(plt)

# Correlation Heatmap
st.write("### Correlation Heatmap")
st.write("""
Korrelationsheatmapet nedenfor viser, hvordan de forskellige variabler i datasættet er korreleret med hinanden. 
Hver celle i heatmappet repræsenterer styrken af sammenhængen mellem to variabler. En høj korrelation (nær +1 eller -1) 
indikerer en stærk sammenhæng mellem variablerne, mens en værdi nær 0 betyder ingen sammenhæng.
""")
# Encode categorical columns
categorical_columns = ['Job_Title', 'Industry', 'Company_Size', 'Location',
                       'AI_Adoption_Level', 'Automation_Risk', 'Required_Skills',
                       'Remote_Friendly', 'Job_Growth_Projection']

df_encoded = df.copy()
le = LabelEncoder()
for col in categorical_columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

corr_matrix = df_encoded.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Heatmap of Correlations in the AI Job Market Dataset")
st.pyplot(plt)
st.write("""
Dette heatmap hjælper med at identificere sammenhænge mellem faktorer som løn, automatiseringsrisiko, 
AI-adoption, og andre variabler i AI-jobmarkedet. Det kan f.eks. afsløre, om højere løn er forbundet med højere eller lavere automatiseringsrisiko.
""")
