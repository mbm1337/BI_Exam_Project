# Design Home Page
# Function to install packages


# Run the function to install packages
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

def install_packages():
    try:
        import pandas
        import numpy
        import seaborn
        import sklearn
        import matplotlib
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

install_packages()

import streamlit as st




banner = """
    <body style="background-color:yellow;">
            <div style="background-color:#385c7f ;padding:10px">
                <h2 style="color:white;text-align:center;">Assessing Job Automation Risk in the AI Era</h2>
            </div>
    </body>
    """

st.markdown(banner, unsafe_allow_html=True
            )


st.markdown(
    """
    ###
        
    :green[Mustafa & Morten]
    
    ### 
"""
)






