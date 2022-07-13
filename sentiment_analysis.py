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