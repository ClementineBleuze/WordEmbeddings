# Project brief
The goal of this project is an attempt to identify dimensions of word embeddings that could potentially encode target linguistic information.
For this we propose a framework consisting of a metric allowing to compare information encoding quality between different models (`InfEnc`) and 
finding dimension candidates that are most like to encode target linguistic information (`stable dimensions`).

For the models, we only chose those BERT-like models working with the French laguage:
- FlauBERT (small cased, based uncased, base cased, large cased)
- CamemBERT (base cased)
- XML-R (base cased, large cased)
- mBERT (base cased, base uncased)
- DistilBERT (base cased)

For the target linguistic features, we used existing resources.
From [Morphalou3](https://www.ortolang.fr/market/lexicons/morphalou):
- Grammatical gender of nouns and adjectives
- Grammatical number of nouns and adjectives
- Part of speech information for nouns, adjectives, and verbs

From [Sequoia](https://github.com/FrSemCor/FrSemCor/blob/master/sequoia-9.1.frsemcor):
- Semantic category Act and Person for nouns

# Repository structure
1. Retrieving word embeddings of the pre-trained models (for models trained on [French data only](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/0a%20-%20Building_corpora.ipynb), for [multilingual models](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/0b%20-%20Building_more_corpora.ipynb)).
2. Comparing tokenization of the same words by [different models](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/0c%20-%20Adding%20semantic%20information.ipynb).
3. InfEnc calculation for the gender feature (for [nouns only](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/1a%20-%20Gender%20experiment%2C%20NOUN.ipynb), for [adjectives only](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/1b%20-%20Gender%20experiment%2C%20ADJ.ipynb), for [nouns and adjectives combined](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/1c%20-%20Gender%20experiment%2C%20NOUN%26ADJ.ipynb)).
4. InfEnc calculation for the number feature (for [nouns only](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/2a%20-%20Number%20experiment%2C%20NOUN.ipynb), for [adjectives only](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/2b%20-%20Number%20experiment%2C%20ADJ.ipynb), for [nouns and adjectives combined](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/2c%20-%20Number%20experiment%2C%20NOUN%26ADJ.ipynb)).
5. InfEnc calculation for the PoS feature (for [nouns](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/3a%20-%20POS%20experiment%2C%20NOUN.ipynb), for [adjectives](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/3b%20-%20POS%20experiment%2C%20ADJ.ipynb), for [verbs](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/3c%20-%20POS%20experiment%2C%20VERB.ipynb)).
6. InfEnc calculation for the semantic category of nouns (for [Person](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/4b%20-%20Semantic%20experiment%2C%20Person.ipynb), for [Act](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/4a%20-%20Semantic%20experiment%2C%20Act.ipynb)).
7. Number of obtained [stable dimensions](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Final%20Experiments/5%20-%20Stable%20dimensions.ipynb) for each feature and the intersections between different features.

# Publication
Ekaterina Goliakova, David Langlois. 2024. [What do BERT word embeddings learn about the French language?](https://github.com/ClementineBleuze/WordEmbeddings/blob/main/Reports/What%20do%20BERT%20word%20embeddings%20learn%20about%20the%20French%20language%3F.pdf). In *Proceedings of the 6th International Conference on Computational Linguistics in Bulgaria (CLIB 2024)*, Bulgaria. Department of Computational Linguistics, IBL -- BAS.
