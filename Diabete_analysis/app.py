
import streamlit as st 
import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 


import pickle 
import base64

st.set_page_config(
    page_title='diabete',
    page_icon='::',
    initial_sidebar_state='auto',
    layout = 'wide'
)

@st.cache_data
def load_data(dataset) :
    data = pd.read_csv(dataset)
    return data

data = load_data('test_data.csv')


# Fonction de telecahrgement d'un fichier 

def filedownload(df) :
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<h1 href ="data:/file/csv, base64, {b64}"> Download a file </h1>'
    return href 

st.sidebar.image('JPEG image-4A09-ABD4-DB-0.jpeg', use_container_width=True)

def main() :
    
    st.markdown('<h1 style="text-align:center; color:brown"> Streamlit Diabetis App </h1>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color: white;'> Diabetis study in Cameroun </h2>",unsafe_allow_html=True)
    
    with st.sidebar :
        menu = ["Home","Analysis","Data visualisation ", "Machine Learning"]

        pages = {
            '🏠 Home' : 'Home',
            '📈 Analysis' : 'Analysis',
            '🗂️ Summary' : 'Summary',
            '📊 Data visualisation' : 'Data_visualisation',
            '⚙️ Machine Learning' : 'Machine_learning'
        }
        
        if 'page' not in st.session_state:
            st.session_state['page'] = 'Home'
        
        for page_name , page_key in pages.items() :
            if( st.button(page_name, key=page_key, use_container_width=True)) :
                st.session_state['page'] = page_key
                

    left, middle, rigth = st.columns((2,3,2))
    if st.session_state['page'] == 'Home' :
        with middle :
            st.image('JPEG image-4A09-ABD4-DB-1.jpeg')
            st.write('"This is an app that will analyse diabetes Datas with some python tools that can optimize decisions"')
            st.subheader('Diabete information ')
            st.write('n Cameroon, the prevalence of diabetes in adults in urban areas is currently estimated at 6 – 8%, with as much as 80% of people living with diabetes who are currently undiagnosed in the population. Further, according to data from Cameroon in 2002, only about a quarter of people with known diabetes actually had adequate control of their blood glucose levels. The burden of diabetes in Cameroon is not only high but is also rising rapidly. Data in Cameroonian adults based on three cross-sectional surveys over a 10-year period (1994–2004) showed an almost 10-fold increase in diabetes prevalence.')

    elif st.session_state['page'] == 'Analysis' :
        st.subheader('Diabete analysis')
        st.write(data)
        
        if st.checkbox('Summary'):
            st.write(data.describe())
        elif st.checkbox('Diabete Corelation'):
            st.write(data.corr())
        elif st.checkbox('Correlation visualisation'):
            fig = plt.figure(figsize=(20,20))
            sns.heatmap(data.corr(), annot=True)
            st.pyplot(fig)
        
    elif st.session_state['page'] == 'Data_visualisation' :
        
        if st.checkbox('Countplot') :
            st.subheader('Countplot')
            fig1 = plt.figure(figsize=(20,20))
            sns.countplot(x='Age', data=data)
            st.pyplot(fig1)
        
        if st.checkbox('Scatterplot') :
            fig2 = plt.figure(figsize=(20,20))
            sns.scatterplot(x='Age', y='glucose', data=data, hue='Outcome')
            st.pyplot(fig1)
            
    elif st.session_state['page'] == 'Machine_learning' :
       
        tab1, tab2, tab3 =  st.tabs([":clipboard: Data", ":bar_chart: Visualisation", ":mask: :smile: Prediction"])

        with tab1 :
                st.subheader('loaf dataset')
                st.write(data)
    
        with tab2 :
                st.subheader(' Histogram Glucose')
                fig =  plt.figure(figsize=(8,8))
                sns.histplot(x='Glucose',data=data)
                st.pyplot(fig)
    
        with tab3 :
            st.subheader('Prediction')
            model = pickle.load(open('model_dump.pkl', 'rb'))
            prediction = model.predict(data)
            pp =pd.DataFrame(prediction, columns=['Prediction'])
            
            df = pd.concat([data, pp], axis=1)
            st.write(df)
            if st.button('Download') :
                st.markdown(filedownload, unsafe_allow_html=True)                
                
if __name__ == '__main__' :
    main()
