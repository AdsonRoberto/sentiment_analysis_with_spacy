!pip install spacy==2.2.3

!python3 -m spacy download pt

import pandas
import string
import spacy
import random
import seaborn as saindo
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

base_dados = pd.read_csv('/content/base_treinamento.txt', encoding = 'utf-8')
base_dados.shape
base_dados.head()
base_dados.tail()
sns.countplot(base_dados['emocao'], label = 'Contagem')

pontuacoes = string.punctuation
pontuacoes

from spacy.lang.pt.stop_words import STOP_WORDS
stop_words = STOP_WORDS

print(stop_words)

len(stop_words)

pln = spacy.load('pt')

def preprocessamento(texto):
  texto = texto.lower()
  documento = pln(texto)
  
  lista = []
  for token in documento:
    #lista.append(token.text)
    lista.append(token.lemma_)

  lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
  lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

  return lista