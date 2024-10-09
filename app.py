# Design Home Page
# Function to install packages
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

# Run the function to install packages
install_packages()
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

try:
    import pandas
    import numpy
    import seaborn
    import sklearn
    import matplotlib
except ImportError:
    print("Installing required packages...")
    os.system("pip install pandas numpy seaborn scikit-learn matplotlib")

from PIL import Image
logo = Image.open('./media/logo.png')

st.set_page_config(
    page_title="Streamlit BI Demo",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:tdi@cphbusiness.dk',
        'About': "https://docs.streamlit.io"
    }
)

st.sidebar.header("Try Me!", divider='rainbow')
# st.sidebar.success("Select a demo case from above")
st.image(logo, width=200)

banner = """
    <body style="background-color:yellow;">
            <div style="background-color:#385c7f ;padding:10px">
                <h2 style="color:white;text-align:center;">Streamlit BI Demo App</h2>
            </div>
    </body>
    """

st.markdown(banner, unsafe_allow_html=True)


st.markdown(
    """
    ###
        
    👈 :green[Select a demo case from the sidebar to experience some of what Streamlit can do for BI!]
    
    ### To learn more
    - Check out [Streamlit Documentation](https://docs.streamlit.io)
    - Contact me by [email](mailto://tdi@cphbusiness.dk)
"""
)






