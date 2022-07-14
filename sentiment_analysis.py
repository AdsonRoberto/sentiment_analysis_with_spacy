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

teste = preprocessamento('Estou aPrendendo 1 10 23 processamento de linguagem natural, Curso em Curitiba')
teste

base_dados['texto'] = base_dados['texto'].apply(preprocessamento)

base_dados.head(10)

exemplo_base_dados = [["este trabalho é agradável", {"ALEGRIA": True, "MEDO": False}],
                      ["este lugar continua assustador", {"ALEGRIA": False, "MEDO": True}]]

type(exemplo_base_dados)

exemplo_base_dados[0]

exemplo_base_dados[0][0]

exemplo_base_dados[0][1]

type(exemplo_base_dados[0][1])

base_dados_final = []
for texto, emocao in zip(base_dados['texto'], base_dados['emocao']):
  #print(texto, emocao)
  if emocao == 'alegria':
    dic = ({'ALEGRIA': True, 'MEDO': False})
  elif emocao == 'medo':
    dic = ({'ALEGRIA': False, 'MEDO': True})

  base_dados_final.append([texto, dic.copy()])

len(base_dados_final)

base_dados_final[0]

base_dados_final[0][0]

base_dados_final[0][1]

type(base_dados_final[0][1])

modelo = spacy.blank('pt')
categorias = modelo.create_pipe("textcat")
categorias.add_label("ALEGRIA")
categorias.add_label("MEDO")
modelo.add_pipe(categorias)
historico = []

modelo.begin_training()
for epoca in range(1000):
  random.shuffle(base_dados_final)
  losses = {}
  for batch in spacy.util.minibatch(base_dados_final, 30):
    textos = [modelo(texto) for texto, entities in batch]
    annotations = [{'cats': entities} for texto, entities in batch]
    modelo.update(textos, annotations, losses=losses)
  if epoca % 100 == 0:
    print(losses)
    historico.append(losses)

modelo = spacy.blank('pt')
categorias = modelo.create_pipe("textcat")
categorias.add_label("ALEGRIA")
categorias.add_label("MEDO")
modelo.add_pipe(categorias)
historico = []

historico_loss = []
for i in historico:
  historico_loss.append(i.get('textcat'))

modelo.to_disk("modelo")