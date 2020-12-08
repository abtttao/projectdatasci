import streamlit as st
import pandas as pd
import pickle



# st.write("""
# ### Hello!
# """)
st.image('./header.png', width=900)

st.sidebar.image('./logo.png', width=210)
st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')


def get_input():
    #widgets

    t_Sex = st.sidebar.radio('Sex', ['Male','Female'])
    t_StudentType = st.sidebar.radio('StudentType', ['THAI','FOREIGN'])
    t_FacultyName = st.sidebar.selectbox('FacultyName', ['School of Agro-industry','School of Cosmetic Science','School of Dentistry','School of Health Science',
    'School of Information Technology','School of Integrative Medicine','School of Law','School of Liberal Arts' ,'School of Management','School of Nursing','School of Sinology' ,'School of Social Innovation'])

    t_Flag = st.sidebar.selectbox('Flag', ['Australia','Brazil','Cameroon','Bhutan','China', 'France','Indonesia','Korea', 'South Korea','Myanmar', 'Laos', 'Japan','Mali','Taiwan','Bangladesh','United States of America','Thailand', 'United Kingdom of Great Britain and Northern Ireland','NaN'])
    t_Tcas = st.sidebar.selectbox('TCAS', [1,2,3,4,5])
   
    

    if t_Sex == 'Male': t_Sex = 'M'
    else: t_Sex = 'F'

    if  t_StudentType == 'THAI':  t_StudentType = '1'
    else:  t_StudentType = '2'

    #dictionary
    data = {'Sex': t_Sex,
            'StudentType': t_StudentType,
            'FacultyName': t_FacultyName,
            'Flag': t_Flag,
            'TCAS': t_Tcas}
            
            
           
                     
    #create data frame
    data_df = pd.DataFrame(data, index=[0]) 
    return data_df

df = get_input()
st.write(df)

data_sample = pd.read_csv('tcas_new.csv')
df = pd.concat([df, data_sample],axis=0)

cat_data = pd.get_dummies(df[['Sex', 'FacultyName', 'Flag']])




#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)
#Drop un-used feature
X_new = X_new.drop(columns=['Sex', 'FacultyName', 'Flag'])



# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
st.write(prediction)

# st.image('./footer.png' , width=950)