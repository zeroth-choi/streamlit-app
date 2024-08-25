import numpy as np
import pandas as pd 
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

iris_dataset = load_iris()

df= pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
df.columns= [ col_name.split(' (cm)')[0] for col_name in df.columns] # 컬럼명 뒤의 cm 제거
df['species']= iris_dataset.target 

species_dict = {0 :'setosa', 1 :'versicolor', 2 :'virginica'} 

def map_species(x):
  return species_dict[x]

df['species'] = df['species'].apply(map_species)

st.sidebar.title('붓꽃(iris)의 종')

# multiselect를 이용하여 여러개 선택 
select_species = st.sidebar.multiselect(
    '확인하고자 하는 종을 선택해 주세요. (복수 선택 가능)',
    ['setosa', 'versicolor', 'virginica']
)

# 라디오에 선택한 내용을 select_radio 변수에 저장
select_radio =st.sidebar.radio(
    "범위를 지정할 기준",
    ['sepal length', 'sepal width', 'petal length', 'petal width'],
    horizontal=True
    )
# 선택한 컬럼의 값의 범위를 지정할 수 있는 slider 생성. 
slider_range = st.sidebar.slider(
    "선택한 기준의 범위",
     0.0,   #시작 값 
     10.0, #끝 값  
    (2.5, 7.5) # 기본 값 범위
)

# 필터 적용 버튼 생성 
start_button = st.sidebar.button(
    "filter 적용"   #"버튼에 표시될 내용"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/220px-Kosaciec_szczecinkowaty_Iris_setosa.jpg', caption="setosa")
with col2:
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/220px-Iris_versicolor_3.jpg', caption="versicolor")
with col3:
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/220px-Iris_virginica.jpg', caption="virginica")
st.markdown('왼쪽에서 **확인하고자 하는 종**과 **범위를 지정할 기준**을 선택한 뒤 **선택한 기준의 범위**를 지정하고 **filter 적용** 버튼을 누르면 그에 해당하는 데이터만 추출하여 표, 도구분포, 산점도를 생성합니다.')

if start_button:
    tmp_df = df[df['species'].isin(select_species)]
    #slider input으로 받은 값을 기준으로 데이터를 필터링
    tmp_df= tmp_df[ (tmp_df[select_radio] >= slider_range[0]) & (tmp_df[select_radio] <= slider_range[1])]
    st.header("선택된 데이터")
    st.dataframe(tmp_df)
  
    st.header("도수분포")
    fig = plt.figure(figsize = (10, 9))
    plt.subplot(2,2,1)
    plt.hist(data=tmp_df, x='sepal length', bins=10, rwidth=0.8)
    plt.xlabel('sepal length (cm)')
    plt.subplot(2,2,2)
    plt.hist(data=tmp_df, x='sepal width', bins=10, rwidth=0.8)
    plt.xlabel('sepal width (cm)')
    plt.subplot(2,2,3)
    plt.hist(data=tmp_df, x='petal length', bins=10, rwidth=0.8)
    plt.xlabel('petal length (cm)')
    plt.subplot(2,2,4)
    plt.hist(data=tmp_df, x='petal width', bins=10, rwidth=0.8)
    plt.xlabel('petal width (cm)')
    st.pyplot(fig)

    st.header("산점도")
    fig2 = sns.pairplot(tmp_df, diag_kind='kde', # or hist
             hue="species", palette='bright')
    st.pyplot(fig2)

    # 성공 문구 
    st.sidebar.success("Filter 적용됨!")
