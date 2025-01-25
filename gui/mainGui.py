import streamlit as st
import requests


VALID_TAGS = ['ORG-POL', 'ORG', 'PRODUCT', 'ORG-OTH', 'EVENT', 'PERSON', 'GPE', 'FAC']
# INVALID_TAGS = ['0', 'O', '<s>', '</s>']
BAD_WORDS = ['0', 'O', '<s>', '</s>']

NER_COLORS = {
    "ORG-POL": "green",                     
    "ORG": "blue",                        
    "PRODUCT": "blue",                   
    "ORG-OTH": "red",                  
    "EVENT": "orange",                    
    "PERSON": "green",                     
    "GPE": "red",                       
    "FAC": "gray",
}
# st.title("FA Task")
inp, out = st.columns(2)

with inp:
    sentence = st.text_area("sentence", "かつては東京都三鷹市にオフィスを構えていたが、大月のキングレコード職を機に事業を停止。") # default w/ this

if st.button("Submit"):
    # here i am using the previously trained model.
    try:
        response = requests.post("http://localhost:8000/predict", json={"text": sentence})
        response.raise_for_status()  
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
    json_ = response.json()
    output = {}
    fullText = ""
    tokens, tags, words, wordTags = [], [], [], []
    for k, v in json_.items():
        tokens.append(k)
        tags.append(v)
#         print(f"{k} --> {v}")

    current_word = ""
    current_tag = None
    for i, token in enumerate(tokens):
        if token not in BAD_WORDS:
            fullText += token  # for debug.

        if i > 0 and tags[i] == tags[i - 1]:  # Continue the current word
            current_word += token
        else:  #end of word.
            if current_word:
                current_word = current_word.strip()
                current_word = current_word.strip("▁")
                output[current_word] = current_tag
                current_word = ""

            if token not in BAD_WORDS:  # starting a new word
                current_word = token
                current_tag = tags[i]

    if current_word:  # add the last word
        output[current_word] = current_tag
    # out.write(f"success: {output}")
    
    color = "black"
    stringOutput = ""
    for k, v in output.items():
#        print (k, v)
        if (v not in BAD_WORDS):
            stringOutput += f"<span style='color: black;'>:{NER_COLORS[v]}-background[{k}] </span> <span style='background-color: {NER_COLORS[v]}; color: white;'>{v}</span>"
        else:
            stringOutput += f"<span style='color: black;'>{k} </span>"
    out.subheader("Output")
    out.markdown(f"{stringOutput}", unsafe_allow_html=True)

if st.button("Clear"):
    st.session_state.sentence = ""
    # st.session_state.output = ""
    
