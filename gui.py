import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Load the trained model
model = load_model('chatbot_model.keras')

# Load the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents
intents = json.loads(open('intents.json').read())

# Load the words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def send_message():
    user_message = entry.get()
    entry.delete(0, tk.END)

    try:
        # Use the chatbot logic from chatbot.py
        ints = predict_class(user_message)
        res = get_response(ints, intents)

        chat_box.config(state=tk.NORMAL)
        chat_box.insert(tk.END, f"You: {user_message}\n")
        chat_box.insert(tk.END, f"Chappie: {res}\n\n")
        chat_box.config(state=tk.DISABLED)
        chat_box.yview(tk.END)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Create the main window
root = tk.Tk()
root.title("Chappie Chatbot")

# Create and configure the chat box
chat_box = tk.Text(root, height=20, width=50, state=tk.DISABLED)
chat_box.grid(row=0, column=0, padx=10, pady=10)

# Create an entry for user input
entry = tk.Entry(root, width=40)
entry.grid(row=1, column=0, padx=10, pady=10)

# Create a button to send messages
send_button = tk.Button(root, text="Send", command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=10)

# Run the GUI
root.mainloop()
