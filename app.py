import streamlit as st
st.title("Iris Classifier- API")

sl = st.slider('Sepal Length', 4.3, 7.9, 0.5)
sw = st.slider('Sepal Width', 2.0, 4.4, 0.5)
pl = st.slider('Petal Length', 1.0, 6.9, 0.5)
pw = st.slider('Petal Width', 0.1,2.5, 0.5)

from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(iris.data,iris.target)

op = model.predict([[sl,sw,pl,pw]])
op = iris.target_names[op[0]]
st.title(op)











            
         
      
        
          


                           
