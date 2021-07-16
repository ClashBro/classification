# Importing the necessary Python modules.
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
import matplotlib.pyplot as plt
import seaborn as sns
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

features_list = glass_df.columns[:-1]
@st.cache
def prediction(model,feature_list):
   result = model.predict(feature_list)
   if result[0] == 1:
      return 'building windows float processed'
   elif result[0] == 2:
      return 'building windows non float processed'
   elif result[0] == 3:
      return 'vehicle windows float processed'
   elif result[0] == 4:
      return 'vehicle windows non float processed'
   elif result[0] == 5:
      return 'containers'
   elif result[0] == 6:
      return 'tableware'
   else :
      return 'headlamps'




st.title('Glass Type Prediction')
st.sidebar.title('Glass Type Prediction')
if st.sidebar.checkbox('Show Raw Data'):
    st.subheader('Glass Type Dataset')
    st.dataframe(glass_df)

st.sidebar.subheader('Visualisation Selector')
selection = st.sidebar.multiselect('Select type of chart',('correlation heatmap','line chart','area chart','count plot','pie chart','box plot'))
if 'line chart' in selection:
    st.subheader('Line Chart')
    st.line_chart(glass_df)
if 'area chart' in selection:
    st.subheader('Area Chart')
    st.area_chart(glass_df)
st.set_option('deprecation.showPyplotGlobalUse', False)
if 'pie chart' in selection:
    st.subheader('Pie Chart')
    pie_data = glass_df['GlassType'].value_counts()
    plt.pie(pie_data, labels=pie_data.index, autopct='%1.2f%%', startangle=30)
    st.pyplot()
if 'correlation heatmap' in selection:
    st.subheader('Correlation Heatmap')
    sns.heatmap(glass_df.corr(),annot = True)
    st.pyplot()
if 'count plot' in selection:
    st.subheader('Count Chart')
    sns.countplot(x = 'GlassType',data = glass_df)
    st.pyplot()
if 'box plot' in selection:
    st.subheader('Box Chart')
    select2 = st.sidebar.selectbox('Choose Column',['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'])
    sns.boxplot(x = select2 , data = glass_df)
    st.pyplot()

st.sidebar.subheader('Value Selector')
ri = st.sidebar.slider('Refractive Index (RI)',float(glass_df['RI'].min()),float(glass_df['RI'].max()))
na = st.sidebar.slider('Sodium (Na)',float(glass_df['Na'].min()),float(glass_df['Na'].max()))
mg = st.sidebar.slider('Magnesium (Mg)',float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
al = st.sidebar.slider('Aluminium (Al)',float(glass_df['Al'].min()),float(glass_df['Al'].max()))
si = st.sidebar.slider('Silicon (Si)',float(glass_df['Si'].min()),float(glass_df['Si'].max()))
k = st.sidebar.slider('Potassium (K)',float(glass_df['K'].min()),float(glass_df['K'].max()))
ca = st.sidebar.slider('Calcum (Ca)',float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
ba = st.sidebar.slider('Barium (Ba)',float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
fe = st.sidebar.slider('Iron (Fe)',float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))

st.sidebar.subheader('Classifier Selector')
clf = st.sidebar.selectbox('Classifier',('Random Forest Classifier','Logistic Regression','Support Vector Machine'))
if clf == 'Support Vector Machine':
    st.sidebar.subheader('Parameter Selctor')
    ker = st.sidebar.radio('Kernel',('linear','rbf','poly'))
    c = st.sidebar.number_input('C(Error Rate)',1,100,1)
    gam = st.sidebar.number_input('Gamma',1,100,1)
    if st.sidebar.button('Classify'):
        st.subheader('Support Vector Machine')
        svc_model = SVC(kernel = ker, C = c , gamma = gam)
        svc_model.fit(X_train,y_train)
        score = svc_model.score(X_train,y_train)
        test_pred = svc_model.predict(X_test)
        pred = prediction(svc_model,[[ri,na,mg,al,si,k,ca,ba,fe]])
        st.write(f'Accuracy : {score}')
        st.write(f'Glass Type : {pred}')
        plot_confusion_matrix(svc_model,X_test,y_test)
        st.pyplot()
if clf == 'Random Forest Classifier':
    st.sidebar.subheader('Parameter Selection')
    n_est = st.sidebar.number_input('Decision Trees(n_estimators)',100,5000,step = 10)
    m_depth = st.sidebar.number_input('Max Depth',1,100,1)
    if st.sidebar.button('Classify'):
        st.subheader('Random Forest Classifier')
        rfc_model = RandomForestClassifier(n_estimators = n_est, max_depth = m_depth, n_jobs = -1)
        rfc_model.fit(X_train,y_train)
        score = rfc_model.score(X_train,y_train)
        test_pred = rfc_model.predict(X_test)
        pred = prediction(rfc_model,[[ri,na,mg,al,si,k,ca,ba,fe]])
        st.write(f'Accuracy : {score}')
        st.write(f'Glass Type : {pred}')
        plot_confusion_matrix(rfc_model,X_test,y_test)
        st.pyplot()
if clf == 'Logistic Regression':
    st.sidebar.subheader('Parameter Selection')
    c = st.sidebar.number_input('C(ErrorRate)',1,100,1)
    iterations = st.sidebar.number_input('Max Iteration',10,10000,step = 10)
    if st.sidebar.button('Classify'):
        st.subheader('Logistic Regression')
        lg_clf = LogisticRegression(C = c,max_iter = iterations)
        lg_clf.fit(X_train,y_train)
        test_pred = lg_clf.predict(X_test)
        pred = prediction(lg_clf,[[ri,na,mg,al,si,k,ca,ba,fe]])
        st.write(f'Accuracy : {score}')
        st.write(f'Glass Type : {pred}')
        plot_confusion_matrix(lg_clf,X_test,y_test)
        st.pyplot()
