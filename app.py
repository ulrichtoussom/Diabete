import streamlit as st 
import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 

import pickle 
import base64


@st.cache_data
def load_data(dataset):
    df= pd.read_csv(dataset)
    return df

data = load_data('diabetes.csv')

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="diabete_predictions.csv">Download CSV File</a>'
    return href


st.sidebar.image('JPEG image-4A09-ABD4-DB-1.jpeg')


def main():
    
    st.markdown("<h1 style='text-align:center;color: brown;'>Streamlit Diabetis App</h1>",unsafe_allow_html=True) 
    st.markdown("<h2 style='text-align:center;color: black;'> Diabetis study in Cameroun </h2>",unsafe_allow_html=True)
    menu = ["Home","Analysis","Data visualisation ", "Machine Learning"]
    choice = st.sidebar.selectbox('salect a Menu', menu)
    
    left, middle, rigth =  st.columns((2,4,2))
    if choice == 'Home':

        with middle:
            st.image('JPEG image-4A09-ABD4-DB-1.jpeg')
            st.write('"This is an app that will analyse diabetes Datas with some python tools that can optimize decisions"')
            st.subheader('Diabete information ')
            st.write('n Cameroon, the prevalence of diabetes in adults in urban areas is currently estimated at 6 – 8%, with as much as 80% of people living with diabetes who are currently undiagnosed in the population. Further, according to data from Cameroon in 2002, only about a quarter of people with known diabetes actually had adequate control of their blood glucose levels. The burden of diabetes in Cameroon is not only high but is also rising rapidly. Data in Cameroonian adults based on three cross-sectional surveys over a 10-year period (1994–2004) showed an almost 10-fold increase in diabetes prevalence.')
    
    elif choice == 'Analysis':
        st.subheader('Diabete information ')
        st.write(data.head())
        
        if(st.checkbox('Summary')):
            st.write(data.describe())
        elif st.checkbox('Diabete correlation'):
            st.write(data.corr())
        elif st.checkbox('Correlation'):
            fig = plt.figure(figsize=(15,15))
            st.write(sns.heatmap(data.corr(),annot=True))
            st.pyplot(fig)
    elif choice == 'Data visualisation ':
        if(st.checkbox('countplot')):
            fig1 = plt.figure(figsize=(15,15))
            sns.countplot(x='Age',data=data)
            st.pyplot(fig1)
        if(st.checkbox('scatterplot')):
            fig2 = plt.figure(figsize=(15,15))
            sns.scatterplot(x='Glucose',y='Age',data=data,hue='Outcome')
            st.pyplot(fig2)
    elif choice == 'Machine Learning':
        tab1 , tab2 , tab3 = st.tabs([":clipboard: Data", ":bar_chart: visualisation", ":mask: :smile: Prediction"])

        uploaded_file = st.sidebar.file_uploader('upload your input csv file ', type=['csv'])
        
        if uploaded_file :
            df = load_data(uploaded_file)
            
            with tab1 :
                st.subheader('loaf dataset')
                st.write(df)
                
            with tab2 :
                st.subheader(' Histogram Glucose')
                fig =  plt.figure(figsize=(8,8))
                sns.histplot(x='Glucose',data=df)
                st.pyplot(fig)
            
            with tab3 :
                model = pickle.load(open('model_dump.pkl','rb')) 
                predictions = model.predict(df)
                st.subheader('Prediction')
                #Tranformation  de 1 array en dataframe   
                pp =  pd.DataFrame(predictions, columns = ['Prediction'])
                
                #concatenation avec le df d'origine   
                ndf =  pd.concat([df,pp], axis=1)
                st.write(ndf)
                 
                button =  st.button('Download csv')
                if button :
                    st.markdown(filedownload(ndf),unsafe_allow_html=True)
        
        
if __name__ == '__main__':
    main()
