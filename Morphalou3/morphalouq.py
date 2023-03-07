import pandas as pd

def queryf(form:str):
    
    nouns = pd.read_csv("data/csv/all_nouns.csv", dtype=str)
    verbs = pd.read_csv("data/csv/all_verbs.csv", dtype=str)
    adj = pd.read_csv("data/csv/all_adjectives.csv", dtype=str)
    adv = pd.read_csv("data/csv/all_adverbs.csv", dtype=str)
    intj = pd.read_csv("data/csv/all_interjections.csv", dtype=str)
    gwords = pd.read_csv("data/csv/all_grammaticalWords.csv", dtype=str)
    others = pd.read_csv("data/csv/all_noCategory.csv", dtype=str)
    found = False

    # Noun
    if form in list(nouns["lemma"]):
        found = True
        print(f"-'{form}' is a common Noun with following attributes:")
        i = list(nouns["lemma"]).index(form)
        print(f"    -gender: {nouns.loc[i,'gender']}")
        if str(nouns.loc[i, 'invariable']) != "nan":
            print(f"    -invariable form: {nouns.loc[i,'invariable']}")
        if str(nouns.loc[i, 'singular']) != "nan":
            print(f"    -singular form: {nouns.loc[i,'singular']}")
        if str(nouns.loc[i, 'plural']) != "nan":
            print(f"    -plural form: {nouns.loc[i,'plural']}")

    elif form in list(nouns["plural"]):
        found = True
        i = list(nouns["plural"]).index(form)
        print(f"-'{form}' is the inflected plural form of common Noun '{nouns.loc[i, 'lemma']}'.")

    # Verbs
    for col in list(verbs.columns):
        if col != "Unnamed: 0":
            if form in list(verbs[col]):
                found = True
                i = list(verbs[col]).index(form)
                if col == "lemma":
                    print(f"-'{form}' is a verb.")
                else:
                    print(f"-'{form}' is an inflected form of verb '{verbs.loc[i, 'lemma']}' with attributes {col}.")  
    
    # Adjectives
    for col in list(adj.columns):
         if col != "Unnamed: 0":
            if form in list(adj[col]):
                found = True
                i = list(adj[col]).index(form)
                if col == "lemma":
                    print(f"-'{form}' is an adjective.")
                else:
                    print(f"-'{form}' is an inflected form of adjective '{adj.loc[i, 'lemma']}' with attributes {col}.")  
    
    # Adverbs
    if form in list(adv["lemma"]):
        found = True
        i = list(adv["lemma"]).index(form)
        if str(adv.loc[i, "locution"]) == "oui":
            print(f"-'{form}' is an adverb (locution).")
        else:
            print(f"-'{form}' is an adverb.")
    
    # Interjections
    if form in list(intj["lemma"]):
        found = True
        i = list(intj["lemma"]).index(form)
        if str(intj.loc[i, "locution"]) == "oui":
            print(f"-'{form}' is an interjection (locution).")
        else:
            print(f"-'{form}' is an interjection.")
    
    # Grammatical Words
    if form in list(gwords["lemma"]):
        found = True
        print(f"-'{form}' is a grammatical word with following attributes:")
        i = list(gwords["lemma"]).index(form)

        if str(gwords.loc[i, 'grammaticalCategory']) != "nan":
            print(f"    -grammatical category: {gwords.loc[i, 'grammaticalCategory']}")
        
        if str(gwords.loc[i, 'grammaticalSubCategory']) != "nan":
            print(f"    -grammatical subcategory: {gwords.loc[i, 'grammaticalSubCategory']}")
        
        if str(gwords.loc[i, 'locution']) == "oui":
            print(f"    -locution")
    
    else:
        for j in range(1,9):
            if form in list(gwords["inflectedForm_"+str(j)]):
                found = True
                i = list(gwords["inflectedForm_"+str(j)]).index(form)
                print(f"-'{form}' is an inflected form of grammatical word '{gwords[i, 'lemma']}'.")

    # No Category
    if form in list(others["lemma"]):
        found = True
        print(f"-'{form}' is a word with unknown category.")

    if not found:
        print("'{form}' hasn't been found in Morphalou3.")