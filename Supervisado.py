import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Ingresamos los datos
import pandas as pd
# Ingresamos los datos
data = {
    'Distancia': [47,47,47,33,33,33,12,12,12],
    'Clima':[1,2,3,1,2,3,1,2,3],
    'Tiempo': [35,40,50,20,24,28,15,22,25],

}

df=pd.DataFrame(data)

#Definición de variables
x = df[['Distancia','Clima','Tiempo']]
y = df['Tiempo']

#Datos para entrenamiento
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Creación del modelo
modelo = LogisticRegression()
modelo.fit(x_train, y_train)

#Predicciones con el conjunto de pruebas
y_pred = modelo.predict(x_test)

#Evaluación del Modelo
print('Matriz de confusión:')
print(confusion_matrix(y_test, y_pred))


print('\nPrecisión del modelo: ', accuracy_score(y_test, y_pred))

