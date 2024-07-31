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

For the target linguistic features we used existing resources.
From Morphalou3:
- Grammatical gender of nouns and adjectives
- Grammatical number of nouns and adjectives
- Part of speech information for nouns, adjectives, and verbs

From Sequoia:
- Semantic category Act and Person for nouns

# Repository structure
1. Retrieving word embeddings of the pre-trained models (for models trained on French data only, for multilingual models).
2. Comparing tokenization of the same words by different models.
3. InfEnc calculation for gender feature (for nouns only, for adjectives only, for nouns and adjectives combined).
4. InfEnc calculation for number feature (for nouns only, for adjectives only, for nouns and adjectives combined).
5. InfEnc calculation for PoS feature (for nouns, for adjectives, for verbs).
6. InfEnc calculation for semantic category of nouns (for Person, for Act).
7. Number of obtained stable dimensions for each feature and the intersections between different features.

# Publication
Ekaterina Goliakova, David Langlois. 2024. What do BERT word embeddings learn about the French language?. In Proceedings of the 5th International Conference on Computational Linguistics in Bulgaria (CLIB 2022), pages 8–12, Sofia, Bulgaria. Department of Computational Linguistics, IBL -- BAS.
