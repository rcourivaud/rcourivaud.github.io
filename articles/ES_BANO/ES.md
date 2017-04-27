
# Tutoriel : Elastic Search - Python (Confirmé) 

Ce post permet d'expliquer pas à pas comment mettre en place un moteur de recherche sur les adresses françaises avec ElasticSearch et un système de reverse geocoding qui permet de retrouver une adresse en fonction de coordonnées GPS. Les adresses sont extraites de la Base d'Adresse Nationale (BAN). Elle  a été constitué avec la collaboration de la Poste et de l'IGN.

Pré-requis : 
- Bases de Python 
- Système sous Ubuntu (VPS, ou en local)

## Description d'Elastic Search

Elastic search (ES) est un moteur de recherche basé sur une architecture REST (qui peut être accédée par le protocole HTTP). Il utilise le moteur open-source Apache Lucène qui permet de la recherche sémantique en texte intégral. Les données sont stockées sous forme de document JSON (clés:valeurs). Une architecture en clusters scalable, réplicable, très modulable est developpée nativement. Cela permet d'avoir une qualité de service (SLA) maximum, et éviter toutes pertes données. Les clusters peuvent être supprimés et ajoutés, le réseau de noeuds s'auto-adapte pour restituer la meilleure qualité de service possible. 

ES peut être utilisé pour faire de la recherche de texte comme Google par exemple. Dans ce post, nous allons le configurer pour réaliser un autocomplete d'adresses. 

## Mise en place d'ElasticSearch 

### Installation de JAVA 

Mise à jour de l'index des packages
```
sudo apt-get update
```

Installation Java Runtime environnement (JRE)
```
sudo apt-get install default-jre
```
Install JAVA JDK
```
sudo apt-get install default-jdk
```

Ajout de l'ORACLE PPA (Personal Package Archives) à l'index et mise à jour, cela permet d'aller chercher JAVA 8 dans un nouveau répertoire.
```
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
```

Installation de JAVA 8
```
sudo apt-get install oracle-java8-installer
sudo apt-get update
```

### Installation et configuration d'ES

Installation d'elasticsearch
```
sudo apt-get install elasticsearch
```

Configuration du cluster 
```
sudo nano /etc/elasticsearch/elasticsearch.yml
```
Décommenter et modifier ces lignes comme suit
```
cluster.name: MyClusterName
http.host: 0.0.0.0         
http.cors.enabled: true    
http.cors.allow-origin: "*"
http.port: 9200
```

Configuration des paramètres de la JVM (Uniquement si la mémoire RAM du serveur est inférieur à 2Go)
```
sudo nano /etc/elasticsearch/jvm.options
```

Commenter ces deux lignes
```
#-Xms2g
#-Xmx2g
```

### Lancer le service ES 

Le service Elastic Search peut être démarré et stoppé avec les commandes suivantes.
```
sudo -i service elasticsearch start
sudo -i service elasticsearch stop
```

Pour vérifier que le status est bien lancé...
```
sudo service elasticsearch status
```


## Configuration d'ES

### Creation de l'index

Pour créer un index (base de données), il faut configurer le moteur d'indexation pour que les recherches sémantiques soient d'une part efficaces (rapides) et d'autre part correspondent aux attentes de l'application. La configuration se fait par le  biais d'un JSON indiquant toutes les actions effectuées sur les données.  

Dans notre cas, voici le dictionnaire permettant de configurer au mieux l'index pour réaliser les requêtes répondants au cahier de charge.


```python
analyzer  = {
  "settings": {
    "index": {
    #Nombre de clusters sur le réseau
      "number_of_shards": 1,
      "analysis": {
        "analyzer": {
        #On définit ici notre analyzer qui permettra de stocker les données, 
        #Un analyzer est constitué de plusieurs champs ..
          "my_analyzer": {
            # Un tokenizer qui permet de découper le texte en token
            "tokenizer": "my_tokenizer",
            # Des filtres qui transforme le texte en minuscule, les caractères en ASCII, etc.
            "filter": ["lowercase", "asciifolding","elision", 'word_delimiter'],
            # Et des filtres permettant de remplacer certains caractères ou paterns. 
            "char_filter" : ["my_mapping"]
          }
        # Nous pouvons maintenant définir les différents champs "custom" de notre analyzer 
        },"tokenizer": {
            # Le tokenizer et un tokenizer edge_ngram
            "my_tokenizer": {
            # edge_ngram forme des n_grams commencant tous par le début du mot. 
            # je m'apelle -> j, je , je m, je ma, je map, je mapp,je mappe, etc.
              "type": "edge_ngram",
                # Nous pouvons donner des ranges 
                # min_ngram permet de retirer tous les ngram d'une lettre Cf : j , m
              "min_gram": 2,
                # max_ngram permet de bloquer à une certain degrès 
              "max_gram": 20,
                # le tokenizer doit prendre aussi des caractères de découpage. Par défaut, 
                # c'est uniquement les espaces.
              "token_chars": [
                "letter",
                "digit",
                "whitespace",
                "punctuation"
              ]
            }
        # Defintion des filtres
        },"filter": {
            # Filtre d'auto complete repete les prérequis du tokenizer
            "autocomplete": {
                "type": "edge_ngram",
                "min_gram": 2,
                "max_gram": 20,
            },
            # Il est possible de filtrer les StopWords en francais, 
            # De nombreuses langues sont disponibles..
            "my_stop": {
                    "type":"stop",
                    "stopwords":"_french_"
                },
            # Un stemmer permet de ne garder uniquement les radicaux des mots.
            "my_stemmer" : {
                    "type" : "stemmer",
                    "name" : "light_french"
                }, 
            # Un dictionniare de synonymes.
            "my_synonym" : {
                    "type" : "synonym",
                    "synonyms" : [
                        # Cette syntaxe indique de rue est un synonym de avenue et inversement
                        "rue, avenue",
                        # même chose pour boulevard
                        "boulevard, boulvard, blvd"
                        ]
        }
                    
        },
        # Defintion des filtres de caractères
        "char_filter" : {
            # On réalise un mapping caractère par caractère
            "my_mapping" : {
                "type" : "mapping",
                # Ici pour supprimer certains caractères spéciaux
                "mappings" : [
                            # Map le caractère sur vide pour les retirer de l'indexation
                            "-=>",
                            ",=>"
                        ]
            }
        }
        }
    }
  },
    # Enfin pour utiliser au mieux les différentes données. Il faut affecter, 
    # des types au différents champs. 
  "mappings": {
    "addresses": {
      "properties": {
        #Location est un champs comportant des latitudes et longitudes. 
        # Il doit être définit comme un geo_point pour pouvoir réaliser des
        # actions sur les coordonées (distance, box), etc.
        "location": {
            "type": "geo_point"
        # Les deux champs de text devant être indexés
        },"road_name": {
            "type":      "text",
            "analyzer":  "my_analyzer"
        },"road_name_clean": {
            "type":      "text",
            "analyzer":  "my_analyzer"
        },"numero":{
          "type": "integer"
        }
      }
    }
  }
}
```

### Indexation des données. 

Les données concernant les adresses en France sont disponlibles sur le [site](https://bano.openstreetmap.fr/data/) d'OpenStreetMap. ATTENTION  : ce fichier fait plus 1,2Go et est assez long à charger en mémoire.


```python
d = pd.read_csv("./full.csv", header=None) 
```

Pour chaque adresse:
- id (unique) : code_insee + codefantoir + numero
- numero : numéro dans la voie avec suffixe (ex: 1, 1BIS, 1D)
- voie : nom de voie
- code_post : code postal sur 5 caractères
- nom_comm : nom de la commune
- source : OSM = donnée directement issue d'OpenStreetMap, OD = donnée provenant de source opendata locale, O+O = donnée de source opendata enrichie par OpenStreetMap, CAD = donnée directement issue du cadastre, C+O = donnée du cadastre enrichie par OSM (nom de voie par exemple)
- lat : latitude en degrés décimaux WGS84
- lon : longitude en degrés décimaux WGS84

On renomme les colonnes 


```python
d.columns = ["code", "numero", "nom_voie", "code_postal", "commune", "source", "lat", "lon"]
```


```python
d = d[(~d.code_postal.isnull()) & (~d.commune.isnull()) &(~d.nom_voie.isnull())]
```


```python
d.shape
```

Pour récupérer les informations importantes..


```python
clean_postal_code = lambda x: "0" + str(int(x)) if len(str(int(x)))==4 else str(int(x))
clean_insee_code = lambda x: x[0:5]
```

Pour indexer un grand nombre de données, il est conseillé de créer des bulks pour faire simple au lieu d'indexer chaque adresse une par une on créer des agrégations de données qui sont importées en même temps. Cela permet de réduire grandement le temps d'indexation.


```python
def create_bulked_data(df, index_name, doc_type):
    bulk_data = []
    droping = []
    for r, row in df.iterrows():
        try:
            road_name = row.nom_voie + ", " + row.commune
            doc = {
                "numero":row['numero'],
                "nom_voie":row.nom_voie,
                "code_post":row.code_postal,
                "code_insee":row.code_insee,
                "city": row.commune,
                "road_name":road_name,
                "road_name_code":road_name+ " " + row.code_post,
                "location": {
                    
                    "lat":row.lat,
                    "lon":row.lon
                    },
                "road":row.numero + road_name
            }
            data_dict = {
                '_op_type': 'index',
                '_index': index_name,
                '_type': doc_type,
                '_source': doc
            }
            bulk_data.append(data_dict)
        except Exception as e:
            print(e)
    return bulk_data
```

Lorsque les données sont formatées, il suffit de les pousser en utilisant l'API.  


```python
def push_data(client,bulk,index_name):
    helpers.bulk(client=client, actions=bulk)
    client.indices.refresh()
    client.count(index=index_name)
```

Instantiation d'un client permetant d'indexer les données sur les différents clusters. ES permet aussi une authentification simple ainsi que si nécessaire les liens vers les clés publiques du serveurs et de l'entité de certification.


```python
elastic = Elasticsearch(hosts=["your_cluster_ip1","your_cluster_ip2",
                              "your_cluster_ip3""your_cluster_ipN"],
                        http_auth=('user', 'password'))
```

Création de l'index 


```python
index_name = "addresses_autocomplete"
doc_type = "addresses"
elastic.indices.delete(index_name)
elastic.indices.create(index_name, body=analyzer)
```

Pour les fichiers csv volumineux, il est possible de l'ouvrir morceau par morceau et d'indexer chaqu'un d'entre eux. 


```python
for chunk in tqdm(pd.read_csv("./full.csv", chunksize=1e6)):
    chunk_light = chunk[(~chunk.nom_voie.isnull()) & (~chunk.commune.isnull())]
    chunk_light["code_postal"] = chunk_light["code_postal"].apply()
    chunk_light["code_insee"] = chunk_light["code"].apply()
    bd = create_bulked_data(chunk_light, index_name, doc_type)
    push_data(elastic, bd, index_name=index_name)
```

Toutes les données sont maintenant intégrées à la base de données, elles sont directement utilisables.

## Quelques requêtes 

Voici quelques exemples de requêtes que l'on peut utiliser pour :

- Trouver toutes les adresses dans une zone ...


```python
elastic.search(index=index_name, body={"query": {
        "geo_bounding_box": { 
          "location": {
            "top_right": {
              "lat": 46.183325,
              "lon": 4.943478
            },
            "bottom_left": {
              "lat": 46.121845,
              "lon":  4.894726
            }
          }
        }
      }
    }
)
```

- Trouver les informations concernant une rue ...


```python
elastic.search(index=index_name, body={
    "query": {
               "match_phrase_prefix": {
                "road_name": {
                  "query":"27 rue du chemin ver"
                  }
              }
            }
})
```

- Effectuer un reverse geocoding ...


```python
elastic.search(index=index_name, body={
  "sort": [
    {
      "_geo_distance": {
        "location": {
            "lat": 45.923673,
            "lon": 5.361834
          },
        "order":"asc",
        "unit":"km", 
        "distance_type": "plane" 
      }
    }
  ]   
})
```
