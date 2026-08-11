import spacy
nlp = spacy.load("en_core_web_sm")


# def part_of_speech_tagging(text, pos_list = ['NOUN', 'PRON', 'VERB', 'ADJ']):
#     doc = nlp(text)
#     pos_tags = [token.text for token in doc if token.pos_ in pos_list]
#     output = " ".join(pos_tags)
#     return output

def lower_replace_special_char(series):
    output = series.str.lower()
    output = output.str.replace(r'\[.*?\]', '', regex=True) # Remove square brackets and their contents
    output = output.str.replace(r'[^\w\s]', '', regex=True) # Remove special characters
    return output
def token_lemma_stopw_join(text):
    doc = nlp(text)
    norm = [token.lemma_ for token in doc if not token.is_stop]
    output = " ".join(norm)
    return output

def nlp_pipeline(series):
    output = lower_replace_special_char(series)
    output = output.apply(token_lemma_stopw_join)
    return output

if __name__ == "__main__":
    print("This is a text preprocessing module for NLP tasks.")
    