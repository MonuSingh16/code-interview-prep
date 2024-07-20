import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
from PIL import Image
import time

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Loan Default Behaviour")
with st.expander("Project Objective"):
    st.write("""To design, build and deploy a loan default behaviour with data of customers applying for a loan.
    The goal is to predict their defaulting behaviour """)

app_form = st.sidebar.form(key="LoanDefault")
app_form.image("../imgs/FairMoney-logo.jpg")    

# Add model select boxes
model_select = app_form.selectbox(
        "Choose Model:",
        ("Random Forest Classifier", "XGB Classifier", "Gradient Boosting Classifier")
    )
submit_button = app_form.form_submit_button(" Execute ")

# Import train dataset to DataFrame
data = pd.read_csv('../dat/credit.csv')
data.rename(columns={'Unnamed: 0':'observation_id'}, inplace=True)
data.set_index('observation_id', inplace=True)

model_results_df = pd.read_csv("../dat/model_results.csv", index_col=False)

# Drop uniformative columns
data.drop(columns=['telephone','residence_history','months_loan_duration'], axis=1, inplace=True)

if submit_button:
    start = time.perf_counter()
    if model_select == "Random Forest Classifier":
        
        # Create tabs for separation of tasks
        tab1, tab2, tab3 = st.tabs(["🗃 Data", "🔎 All Model Results", "🤓 Model Evaluation"])

        with tab1:    
            # Data Section Header
            st.header("Raw Data")

            # Display first 100 samples of the dateframe
            st.dataframe(data.head(10))

            st.header("Correlations")

            # Heatmap
            corr = data.corr()
            fig = px.imshow(corr)
            st.write(fig)

        with tab2:
            st.header("Model Architecture and Results")

            model_results_df = pd.read_csv("../dat/model_results.csv", index_col=False)
            st.write(model_results_df)

        with tab3:
            st.header("Model Evaluation")

            st.image("../imgs/rfc-confusion-matrix.png")
            st.write(model_results_df[model_results_df['model']=='Random Forest Classifier'])
    
    elif model_select == "XGB Classifier":
        
        # Create tabs for separation of tasks
        tab1, tab2, tab3 = st.tabs(["🗃 Data", "🔎 All Model Results", "🤓 Model Evaluation"])

        with tab1:    
            # Data Section Header
            st.header("Raw Data")

            # Display first 100 samples of the dateframe
            st.dataframe(data.head(10))

            st.header("Correlations")

            # Heatmap
            corr = data.corr()
            fig = px.imshow(corr)
            st.write(fig)

        with tab2:
            st.header("Model Architecture and Results")

            model_results_df = pd.read_csv("../dat/model_results.csv", index_col=False)
            st.write(model_results_df)
        
        with tab3:
            st.header("Model Evaluation")

            st.image("../imgs/xgb-confusion-matrix.png")
            st.write(model_results_df[model_results_df['model']=='XGB Classifier'])

    elif model_select == "Gradient Boosting Classifier":
        
        # Create tabs for separation of tasks
        tab1, tab2, tab3 = st.tabs(["🗃 Data", "🔎 All Model Results", "🤓 Model Evaluation"])

        with tab1:    
            # Data Section Header
            st.header("Raw Data")

            # Display first 100 samples of the dateframe
            st.dataframe(data.head(10))

            st.header("Correlations")

            # Heatmap
            corr = data.corr()
            fig = px.imshow(corr)
            st.write(fig)

        with tab2:
            st.header("Model Architecture and Results")

            model_results_df = pd.read_csv("../dat/model_results.csv", index_col=False)
            st.write(model_results_df)

        with tab3:
            st.header("Model Evaluation")

            st.image("../imgs/gbt-confusion-matrix.png")
            st.write(model_results_df[model_results_df['model']=='Gradient Boosting Classifier'])

    else:
        st.write("No Model Selected")

    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
    st.info('Elapsed time to run through prediction function is %.3f seconds.' % elapsed, icon="ℹ️")
    # The .3f is to round to 3 decimal places.



    
