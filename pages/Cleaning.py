# Importer nødvendige biblioteker
import streamlit as st
import pandas as pd
import numpy as np

# Opret titlen på siden
st.title("Data Preparation & Cleaning Process")

# Load data
df = pd.read_csv('data/ai_job_market_insights.csv')

# Introduktion
st.write("""
### Introduktion til datarensning:
Når vi arbejder med rå data, er der ofte behov for at udføre forskellige trin for at forberede dem til analyse eller modellering.
Dette omfatter identifikation og håndtering af manglende værdier, behandling af outliers (udfaldsdata) samt konvertering af variabler til passende formater.
Nedenfor forklarer vi trin for trin, hvordan datarensningen blev udført i dette projekt.
""")

# Manglende data
st.header("1. Håndtering af manglende data")
st.write("""
Det første trin er at identificere, om der er manglende værdier i datasættet.
Manglende data kan skabe problemer for modeller, så vi skal enten fjerne dem eller udfylde dem med passende værdier (f.eks. medianen eller gennemsnittet).
Her vises en oversigt over de manglende værdier i hver kolonne.
""")

# Check for missing data
missing_data = df.isnull().mean() * 100
st.write("Procentdel af manglende værdier pr. kolonne:")
st.write(missing_data)


# Håndtering af manglende data
st.subheader("Handling af manglende data")
st.code("""
# Håndter NaN-værdier i numeriske kolonner ved at udfylde dem med gennemsnittet
df.fillna(df.mean(), inplace=True)
""", language='python')

# Outliers (udfaldsdata)
st.header("2. Håndtering af outliers")
st.write("""
Outliers er datapunkter, der ligger væsentligt uden for det normale område af dataene. Hvis de ikke håndteres, kan de skævvride analysen.
I dette projekt brugte vi interkvartilområdet (IQR) til at identificere og fjerne outliers.
""")

# Beskrivelse af IQR-metoden
st.subheader("IQR-metoden til at identificere og fjerne outliers")
st.code("""
# Udregn IQR (Interkvartilinterval)
Q1 = df['Salary_USD'].quantile(0.25)
Q3 = df['Salary_USD'].quantile(0.75)
IQR = Q3 - Q1

# Fjern outliers
df = df[~((df['Salary_USD'] < (Q1 - 1.5 * IQR)) | (df['Salary_USD'] > (Q3 + 1.5 * IQR)))]
""", language='python')

st.write("""
Dette script beregner IQR for 'Salary_USD'-kolonnen og fjerner outliers, der falder uden for 1.5 gange IQR-området. 
Dette hjælper med at sikre, at de ekstreme værdier ikke påvirker analysen unødvendigt.
""")

# Konvertering af kolonner
st.header("3. Konvertering af kategoriske variabler til numeriske")
st.write("""
Mange machine learning-modeller kræver, at alle variabler er numeriske. Derfor skal vi konvertere kategoriske variabler til numeriske værdier.
Dette blev gjort ved hjælp af en **Label Encoder**, som kortlægger hver kategori til et heltal.
""")

# Vis kode til label encoding
st.subheader("Label Encoding")
st.code("""
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
categorical_columns = ['Job_Title', 'Industry', 'Company_Size', 'Location', 'AI_Adoption_Level', 
                       'Required_Skills', 'Remote_Friendly', 'Job_Growth_Projection']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))  # Konverter kategorier til numeriske værdier
""", language='python')

st.write("""
Som vist ovenfor blev kategoriske variabler som 'Job_Title' og 'Industry' konverteret til numeriske værdier ved hjælp af Label Encoding. Dette er nødvendigt for at kunne bruge variablerne i modeller som Decision Tree Classifier og Random Forest.
""")

# Resultat af datarensning
st.header("4. Resultat af datarensningsprocessen")
st.write("""
Efter rengøring af dataene ser datasættet nu sådan ud:
""")
st.write(df.head())

# Afslutning
st.write("""
### Afslutning:
Datacleaning er et vigtigt trin i enhver data science-pipeline. Ved at håndtere manglende værdier, outliers og konvertere kategoriske variabler til numeriske,
har vi forberedt vores datasæt til videre analyse og modellering. Dette sikrer, at vi får mere præcise og pålidelige resultater.
""")
