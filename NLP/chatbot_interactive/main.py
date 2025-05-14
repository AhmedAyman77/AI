import nltk
import numpy as np
import random
import json
import tkinter as tk
import arabic_reshaper
import tensorflow as tf
import os
from nltk.stem.lancaster import LancasterStemmer
from bidi.algorithm import get_display

nltk.download('punkt')
stemmer = LancasterStemmer()

def normalize_arabic(text):
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا") 
    text = text.replace("اً", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")

    return text

def reshape_arabic(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        pattern = normalize_arabic(pattern)
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]

training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        bag.append(1 if w in wrds else 0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

model_path = "chatbot_model.h5"

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(training[0]),)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.fit(training, output, epochs=150, batch_size=8)

def bag_of_words(s, words):
    s = normalize_arabic(s)
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def get_response(inp):
    input_data = np.array([bag_of_words(inp, words)])
    results = model.predict(input_data)[0]
    results_index = np.argmax(results)
    tag = labels[results_index]

    if results[results_index] >= 0.8:
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
                return random.choice(responses)
    else:
        return "لم أفهم ما قلت!"

def send():
    user_input = entry_box.get("1.0", tk.END).strip()
    if user_input == "":
        return
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "user: " + reshape_arabic(user_input) + "\n", "user")
    entry_box.delete("1.0", tk.END)

    bot_response = get_response(user_input)
    chat_log.insert(tk.END, reshape_arabic("chat: " + bot_response + "\n"), "bot")
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)

root = tk.Tk()
root.title("مساعد ذكي")

chat_log = tk.Text(root, bg="white", fg="black", font=("Arial", 14))
chat_log.config(state=tk.DISABLED)
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry_box = tk.Text(root, height=3, font=("Arial", 14))
entry_box.pack(padx=10, pady=5, fill=tk.X)

send_button = tk.Button(root, text=reshape_arabic("إرسال"), command=send, font=("Arial", 14))
send_button.pack(padx=10, pady=5)

chat_log.tag_config("user", foreground="blue")
chat_log.tag_config("bot", foreground="green")

root.mainloop()
