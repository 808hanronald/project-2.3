import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import joblib
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import json
with open('config/filepaths.json') as f:
    FPATHS = json.load(f)
st.title("Predicting Movie Raiting from Reviews")

X_to_pred = st.text_input('### Enter text to predict here:', value="I loved the movie and the actors were amazing")

# Loading the ML model
## Two ways of doing it 1st way
@st.cache_resource
def load_ml_model(fpath):
    loaded_model = joblib.load(fpath)
    return loaded_model
# Load model for FPATHS
model_fpath = FPATHS['models']['rf']
rf_pipe = load_ml_model(model_fpath)

# Load Target Lookup dict
## 2nd way
@st.cache_data
def load_lookup(fpath=FPATHS['Data']['ml']['target_lookup']):
    return joblib.load(fpath)

# Load Encoder
@st.cache_resource
def load_encoder(fpath = FPATHS['Data']['ml']['label_encoder']):
    return joblib.load(fpath)

target_lookup = load_lookup()
encoder = load_encoder()

# Prediction Function
def make_prediction(X_to_pred,rf_pipe=rf_pipe, lookup_dict=target_lookup):
    # Get Prediction
    pred_class = rf_pipe.predict([X_to_pred])[0]
    # Decode label
    pred_class = lookup_dict[pred_class]
    pred_class

# Lime Text Explainer
from lime.lime_text import LimeTextExplainer
@st.cache_resource
def get_explainer(class_names = None):
    lime_explainer = LimeTextExplainer(class_names=class_names)
    return lime_explainer

# Explainer
def explain_instance(explainer, X_to_pred, predict_func):
    explanation = explainer.explain_instance(X_to_pred, predict_func)
    return explanation.as_html(predict_proba=False)

# Create the lime Explainer
explainer = get_explainer(class_names = encoder.classes_)

# Obtain an wxplanation for our X_to_pred text
explanation = explain_instance(explainer, X_to_pred,
predict_func = rf_pipe.predict_proba)

# To Display in the notebook
from IPython.display import display, HTML
display(HTML(explanation))


# Trigger prediction and explanation with a button
if st.button('Get Prediction.'):
    pred_class_name = make_prediction(X_to_pred)
    st.markdown(f"#### Predicted Category: {pred_class_name}")
    # Get the Explanation as html and display using the .html component
    html_explanation = explain_instance(explainer, X_to_pred, rf_pipe.predict_proba)
    components.html(html_explanation, height=400)
else:
    st.empty()

# Loading our training and test data
@st.cache_data
def load_Xy_data(joblib_fpath):
    return joblib.load(joblib_fpath)

# Load training data from FPATH
train_data_path = FPATHS['Data']['ml']['train']
X_train, y_train = load_Xy_data(train_data_path)
# Load testing data from FPATH
test_data_path  = FPATHS['Data']['ml']['text']
X_test, y_test = load_Xy_data(FPATHS['Data']['ml']['text'])


def classification_metrics_streamlit(y_true, y_pred, label='',
                           output_dict=False, figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False,values_format=".2f",
                                    class_names=None):
    """Modified version of classification metrics function from Intro to Machine Learning.
    Updates:
    - Reversed raw counts confusion matrix cmap  (so darker==more).
    - Added arg for normalized confusion matrix values_format
    """
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    # Get the classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    ## Print header and report
    header = "-"*70
    ## print(header, f" Classification Metrics: {label}", header, sep='\n')
    ## print(report)
    ## FOR STREAMLET * MAKE JOINED STRING OF HEADER AND REPORT
    final_report = "\n".join([header,f"Classification Metrice: {label}", header,report, "\n"])
    
    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    # Create a confusion matrix  of raw counts (left subplot)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=None, 
                                            cmap='gist_gray_r',# Updated cmap
                                            display_labels=class_names, #ADDED
                                            
                                            values_format="d", 
                                            colorbar=colorbar,
                                            ax = axes[0]);
    axes[0].set_title("Raw Counts")
    
    # Create a confusion matrix with the data with normalize argument 
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=normalize,
                                            cmap=cmap, 
                                            values_format=values_format, 
                                            display_labels=class_names, #ADDED
                                            
                                            colorbar=colorbar,
                                            ax = axes[1]);
    axes[1].set_title("Normalized Confusion Matrix")
    
    # Adjust layout and show figure
    fig.tight_layout()
    ## plt.show()
    
    # Return dictionary of classification_report
    ##if output_dict==True:
    ##     report_dict = classification_report(y_true, y_pred, output_dict=True)
    return final_report,fig


col1,col2,col3 = st.columns(3)
show_train = col1.checkbox("Show training data.", value=True)
show_test = col2.checkbox("Show test data.", value=True)
show_model_params =col3.checkbox("Show model params.", value=False)
if st.button("Show model evaluation."):
    
    if show_train == True:
        # Display training data results
        y_pred_train = rf_pipe.predict(X_train)
        report_str, conf_mat = classification_metrics_streamlit(y_train, y_pred_train, label='Training Data')
        st.text(report_str)
        st.pyplot(conf_mat)
        st.text("\n\n")
    if show_test == True: 
        # Display the trainin data resultsg
        y_pred_test = rf_pipe.predict(X_test)
        report_str, conf_mat = classification_metrics_streamlit(y_test, y_pred_test, cmap='Reds',label='Test Data')
        st.text(report_str)
        st.pyplot(conf_mat)
        st.text("\n\n")
        
    if show_model_params:
        # Display model params
        st.markdown("####  Model Parameters:")
        st.write(rf_pipe.get_params())
else:
    st.empty()






























































