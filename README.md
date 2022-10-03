# Trabajo Práctico NLP

## Fake News Detector
Este trabajo tiene como objetivo crear un clasificador de notificas falsas que sea capaz de detectarlas, con el objetivo de reducir la desinformación en la sociedad sin necesidad de manualmente comprobarlas.
## Librerías Utilizadas
- sklearn
- wordcloud
- pandas
- matplotlib
- nltk

## Guía de Uso
Tener el dataset de ```news.csv``` descargado, moverlo a la carpeta ```src/datasets/```. [Dataset](https://drive.google.com/file/d/10Ese5jJvy98EZhrDDm-41qE6FQnpFIW8/view?usp=sharing).
```sh
python3 -m venv env
source env/bin/activate
cd src
python preprocessing.py
python main.py
```

## License
MIT
