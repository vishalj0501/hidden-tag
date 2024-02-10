import streamlit as st
import spacy
from spacy import displacy
from main import custom_test

nlp = spacy.load("en_core_web_sm")

def pos_tagger(sentence):
    words = sentence.split(" ")
    return custom_test(words)[0]

def main():
    st.title("Parts of Speech Tagger")
    sentence = st.text_input("Enter a sentence:")

    if st.button("Tag"):
        if sentence:
            tags = pos_tagger(sentence)
            st.write("Parts of Speech Tags:")
            st.write(tags)

            doc = nlp(sentence)
            html = displacy.render(doc, style='dep', options={'distance': 100}, page=True)

            st.write(html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
