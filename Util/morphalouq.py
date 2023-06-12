"""Code to query Morphalou3 database for a given word form."""

import pandas as pd


def queryf(form:str, verbose=True, pos_list=False):

    if pos_list:
        pos = set()

    nouns = pd.read_csv("../Data/Morphalou/all_nouns.csv", dtype=str)
    verbs = pd.read_csv("../Data/Morphalou/all_verbs.csv", dtype=str)
    adj = pd.read_csv("../Data/Morphalou/all_adjectives.csv", dtype=str)
    adv = pd.read_csv("../Data/Morphalou/all_adverbs.csv", dtype=str)
    intj = pd.read_csv("../Data/Morphalou/all_interjections.csv", dtype=str)
    gwords = pd.read_csv("../Data/Morphalou/all_grammaticalWords.csv", dtype=str)
    others = pd.read_csv("../Data/Morphalou/all_noCategory.csv", dtype=str)
    found = False

    # Noun
    if form in list(nouns["lemma"]):
        found = True

        if pos_list:
            pos.add('Noun')

        i = list(nouns["lemma"]).index(form)
        if verbose:
            print(f"-'{form}' is a common Noun with following attributes:")
            print(f"    -gender: {nouns.loc[i,'gender']}")
        if str(nouns.loc[i, 'invariable']) != "nan" and verbose:
            print(f"    -invariable form: {nouns.loc[i,'invariable']}")
        if str(nouns.loc[i, 'singular']) != "nan" and verbose:
            print(f"    -singular form: {nouns.loc[i,'singular']}")
        if str(nouns.loc[i, 'plural']) != "nan" and verbose:
            print(f"    -plural form: {nouns.loc[i,'plural']}")

    elif form in list(nouns["plural"]):
        if pos_list:
            pos.add('Noun')

        found = True
        i = list(nouns["plural"]).index(form)
        if verbose:
            print(f"-'{form}' is the inflected plural form of common Noun '{nouns.loc[i, 'lemma']}'.")

    # Verbs
    for col in list(verbs.columns):
        if col != "Unnamed: 0":
            if form in list(verbs[col]):
                if pos_list:
                    pos.add('Verb')

                found = True
                i = list(verbs[col]).index(form)
                if col == "lemma" and verbose:
                    print(f"-'{form}' is a verb.")
                elif verbose:
                    print(f"-'{form}' is an inflected form of verb '{verbs.loc[i, 'lemma']}' with attributes {col}.")  
    
    # Adjectives
    for col in list(adj.columns):
         if col != "Unnamed: 0":
            if form in list(adj[col]):
                found = True
                if pos_list:
                    pos.add('Adjective')

                i = list(adj[col]).index(form)
                if col == "lemma" and verbose:
                    print(f"-'{form}' is an adjective.")
                elif verbose:
                    print(f"-'{form}' is an inflected form of adjective '{adj.loc[i, 'lemma']}' with attributes {col}.")  
    
    # Adverbs
    if form in list(adv["lemma"]):
        if pos_list:
            pos.add('Adverb')

        found = True

        i = list(adv["lemma"]).index(form)
        if str(adv.loc[i, "locution"]) == "oui" and verbose:
            print(f"-'{form}' is an adverb (locution).")
        elif verbose:
            print(f"-'{form}' is an adverb.")
    
    # Interjections
    if form in list(intj["lemma"]):
        if pos_list:
            pos.add('Interjection')

        found = True

        i = list(intj["lemma"]).index(form)
        if str(intj.loc[i, "locution"]) == "oui" and verbose:
            print(f"-'{form}' is an interjection (locution).")
        elif verbose:
            print(f"-'{form}' is an interjection.")
    
    # Grammatical Words
    if form in list(gwords["lemma"]):
        if pos_list:
            pos.add('GW')

        found = True

        if verbose:
            print(f"-'{form}' is a grammatical word with following attributes:")
        i = list(gwords["lemma"]).index(form)

        if str(gwords.loc[i, 'grammaticalCategory']) != "nan" and verbose:
            print(f"    -grammatical category: {gwords.loc[i, 'grammaticalCategory']}")
        
        if str(gwords.loc[i, 'grammaticalSubCategory']) != "nan" and verbose:
            print(f"    -grammatical subcategory: {gwords.loc[i, 'grammaticalSubCategory']}")
        
        if str(gwords.loc[i, 'locution']) == "oui" and verbose:
            print(f"    -locution")
    
    else:
        for j in range(1,9):
            if form in list(gwords["inflectedForm_"+str(j)]):
                if pos_list:
                    pos.add('GW')

                found = True

                i = list(gwords["inflectedForm_"+str(j)]).index(form)
                if verbose:
                    print(f"-'{form}' is an inflected form of grammatical word '{gwords[i, 'lemma']}'.")

    # No Category
    if form in list(others["lemma"]):
        if pos_list:
            pos.add('Unknown')

        found = True

        if verbose:
            print(f"-'{form}' is a word with unknown category.")

    if not found and verbose:
        print("'{form}' hasn't been found in Morphalou3.")

    if pos_list:
        return list(pos)