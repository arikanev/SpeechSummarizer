#### POPUP WINDOW ####
# NOTE - still doens't work in colab or jupuyter
# Copy and paste this into it's own python file

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import tkinter as tk
import requests
import nltk
import os
import openai

import speech_recognition as sr


from bs4 import BeautifulSoup
from tkinter import filedialog


from transformers import pipeline

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


def empirical_comparison(string=""):
  lsa_sentences = run_lsa(string)
  T5_sentences = run_T5(string)
  GPT4_sentences = run_GPT4(string)
  return({"LSA_summarizer":lsa_sentences, "T5_summarizer":T5_sentences, 
"GPT-4_summarizer":GPT4_sentences})
 
def run_GPT4(string):
  #Scott's API key (...shh its private)
  openai.api_key = ""

  response = openai.Completion.create(
    model="text-davinci-003",
    prompt="what color is the sky and why?",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "assistant", "content": string}
    ]
  )
  output = completion.choices[0].message
  return output['content']

def run_T5(string):
	T5_summarizer = pipeline(task="summarization", 
model="scottn66/text-summarization")
	return T5_summarizer(string)


def run_lsa(string):
	nltk.download('punkt')

	LANGUAGE = 'english'
	SENTENCES_COUNT = 2
	lsa_sentences = []

# url = 'https://en.wikipedia.org/wiki/Automatic_summarization'
# parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
# or plain text files
# parser = PlaintextParser.from_file("document.txt", Toeknizer(LANGUAGE))
	parser = PlaintextParser.from_string(string, Tokenizer(LANGUAGE))
	stemmer = Stemmer(LANGUAGE)

	LSA_summarizer = Summarizer(stemmer)
	LSA_summarizer.stop_words = get_stop_words(LANGUAGE)

	for sentence in LSA_summarizer(parser.document, SENTENCES_COUNT):
		lsa_sentences.append(sentence)
	return lsa_sentences




def type_text():
    global text_to_sum
    # Function for the "Type or paste your own text" button
    text = text_box.get("1.0", "end-1c")
    root.destroy()

    text_to_sum = text
    return text

def choose_file():
    global text_to_sum
    # Function for the "Choose text from a local .txt file" button
    file_path = filedialog.askopenfilename()
    with open(file_path, "r") as file:
        text = file.read()
    print(text)
    text_to_sum = text
    root.destroy()
    return


def choose_link():
    global text_to_sum
    # Function for the "Choose text from an internet link" button
    link = link_box.get()
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    print(text)
    text_to_sum = text
    root.destroy()
    return

def record_to_text():

    r=sr.Recognizer()

    r.non_speaking_duration = 2

    r.pause_threshold = 2

    m=sr.Microphone()

    GOOGLE_CLOUD_SPEECH_CREDENTIALS = "macro-atom-182501-f4b1f0237c28.json"

    with m as source:
        print("Please state what you intend to be summarized.")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        try:
            spoken_answer = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS).lower().strip()
            # spoken_answer = r.recognize_google(audio).lower()
            # spoken_answer = r.recognize_sphinx(audio, keyword_entries=[(prompt[idx], 0.8)]).lower().strip()
        except sr.UnknownValueError:
            spoken_answer = ""

    return spoken_answer



global root
global text_box
global link_box

# Set up the tkinter window and widgets
root = tk.Tk()
root.title("Choose your Text for Summary")

text_label = tk.Label(root, text="Type or paste your own text:")
text_label.pack()

text_box = tk.Text(root)
text_box.pack()

submit_button = tk.Button(root, text="Submit", command=type_text)
submit_button.pack()

file_label = tk.Label(root, text="Choose text from a local .txt file:")
file_label.pack()

file_button = tk.Button(root, text="Choose file", command=choose_file)
file_button.pack()

link_label = tk.Entry(root, text="Choose text from an internet link:")
link_label.pack()

link_box = tk.Entry(root)
link_box.pack()

link_button = tk.Button(root, text="Choose link", command=choose_link)
link_button.pack()

root.mainloop()


# print(empirical_comparison(text_to_sum))

print(empirical_comparison(record_to_text()))
