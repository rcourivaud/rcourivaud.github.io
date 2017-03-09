

```python
import pandas as pd 
from sklearn.datasets import load_iris
import seaborn as sns
%matplotlib inline
```

Le but de cet article est de montrer comment on peut mettre en place une API permettant d'utiliser un modèle. Pour cela nous allons mettre en place une classification très facile avec un jeu de données utilisé par tous les jeunes Data -Scientists permettant de comprendre les bases de l'annalyse de données. CE jeu de données est constitué des 150 échantillons de fleurs. On possède les caractéristiques physiques de ces fleurs et le but du jeu est de trouver le type de fleur en fonction de ces derniers.

# Chargement des données et rapides annalyses 


```python
iris = load_iris()
X = iris.data
Y = iris.target
```


```python
X.shape, iris.feature_names
```




    ((150, 4),
     ['sepal length (cm)',
      'sepal width (cm)',
      'petal length (cm)',
      'petal width (cm)'])




```python
df_iris = pd.DataFrame(X, columns=iris.feature_names)
df_iris["type"] = Y
```


```python
sns.violinplot(x="type", y= 'sepal length (cm)', data = df_iris);
```


![png](output_6_0.png)



```python
sns.violinplot(x="type", y= 'petal length (cm)', data = df_iris);
```


![png](output_7_0.png)


On peut déja voir dans un premier temps que ces simples indicateurs permettent, à l'oeil nu de distinguer les différentes espèces.


```python
sns.pairplot(df_iris, hue="type")
```




    <seaborn.axisgrid.PairGrid at 0x1bf31321dd8>




![png](output_9_1.png)


Ces graphs permettent de très bien faire la différences entre les différentes espèces de plantes.

# Modèle 

Pour notre example nous allons utiliser uniquement une Regression Logistique pour avoir de bonnes performances dans des temps très corrects.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
```


```python
ShuffleSplit(1)
```




    ShuffleSplit(n_splits=1, random_state=None, test_size=0.1, train_size=None)




```python
X_train, X_test, y_train, y_test = train_test_split(df_iris.drop("type", axis=1),Y)
```


```python
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
lr_model.predict(X_test)
```




    array([1, 2, 1, 2, 0, 2, 2, 2, 0, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0,
           2, 2, 0, 0, 2, 1, 0, 2, 0, 2, 1, 1, 0, 0, 0])



Même si dans ce cas l'apprentissage du modèle est très rapide il vaut mieux stocker (dumper) en mémoire notre modèle afin d'uniquement le recharger lorsque l'on en a besoin. Pour cela, sklearn mets en place un module joblib qui permet de remplir exactement ce rôle.


```python
from sklearn.externals import joblib
```


```python
joblib.dump(lr_model, "./logisticregression.pkl")
```




    ['./logisticregression.pkl']



Pour le recharger il suffit de faire un load de la même manière ...

# API Flask

On veut rendre disponible notre modèle sur le réseau, pour cela on utilise Flask qui permet de créer très facilement une API en Python. Pour cela il suffit de créer un fichier api.py. 

Si vous n'avez jamais touché à une API, les route sont des points de l'url. Ils permettent de structurer une api. Nous avons donc fait un point d'API `predict_species` qui permet de récupérer les données et de prédire les espèces de plantes. 


```python
from flask import Flask, request
import pandas as pd
from sklearn.externals import joblib
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'MAchine Learning API !'

@app.route('/predict_species' , methods=['POST'])
def predict_species():
    if request.method == 'POST':
        dataframe = pd.DataFrame(request.get_json()["data"])
        model = joblib.load("./logisticregression.pkl")
        return json.dumps({"result":[int(elt) for elt in list(model.predict(dataframe))]})


if __name__ =="__main__":
    app.run()

```

     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    

Les API en python sont très faciles à mettre en place, en évitant les problèmes d'encodage, il faut que les résultats soient serialiables (peuvent être transformés en string). Dans notre cas, les numpy array et les int64 de numpy ne le sont pas. C'est pour cela qu'il faut les convertir en objets de base python pour pouvoir les transportés. 


```python
import json, requests
```


```python
headers = {'Content-Type' : 'application/json'}
```

Il nous suffit alors d'envoyer un POST sur le point d'API créé, avec les données de test en data. Cela nous renvoit les données prédites. 


```python
response = requests.post("http://localhost:5000/predict_species", 
                         headers=headers, 
                         data=json.dumps({"data":[{iris.feature_names[i]:d for i,d in enumerate(elt)} for elt in X_test.as_matrix()], 
                                         "columns":list(X_test.columns)}))
```


```python
response
```




    <Response [200]>




```python
response.text
```




    '{"result": [1, 2, 1, 2, 0, 2, 2, 2, 0, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0, 2, 2, 0, 0, 2, 1, 0, 2, 0, 2, 1, 1, 0, 0, 0]}'




```python
y_test
```




    array([1, 2, 1, 2, 0, 2, 2, 2, 0, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 0,
           1, 1, 0, 0, 2, 1, 0, 2, 0, 1, 1, 1, 0, 0, 0])




```python
from sklearn.metrics import confusion_matrix
```


```python
confusion_matrix(y_test, response.json()["result"])
```




    array([[10,  0,  0],
           [ 0, 12,  3],
           [ 0,  0, 13]])


