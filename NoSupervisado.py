import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Crear el dataset
data = {
    'Distancia': [47,47,47,33,33,33,12,12,12],
    'Clima':[1,2,3,1,2,3,1,2,3],
    'Tiempo': [35,40,50,20,24,28,15,22,25],

}

df=pd.DataFrame(data)

# Seleccionar las columnas relevantes para el clustering
x=df[['Distancia','Tiempo','Clima']].values


# Aplicar K-Means con k=3 (puedes ajustar k según tu necesidad)
kmeans = KMeans(n_clusters=3, random_state=42).fit(x)
df['Cluster']=kmeans.labels_

# Mostrar resultados
print(df)

# Visualizar los clusters
plt.scatter(df['Tiempo'], df['Distancia'], cmap='viridis')
plt.title('Clustering de Distancias y Tiempo')
plt.xlabel('Tiempo')
plt.ylabel('Distancia')
plt.show()