import numpy as np
import os
from collections import Counter

def preprocess_text(text):
    text = ''.join(filter(str.isalpha, text.lower()))
    letter_counts = Counter(text)
    total_letters = sum(letter_counts.values())
    vector = np.array([letter_counts.get(chr(i), 0) / total_letters for i in range(ord('a'), ord('z')+1)])
    return vector

def train_network(data_path, learning_rate=0.01, epochs=10):

    languages = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
    num_languages = len(languages)
    weights = np.random.rand(num_languages, 26)

    for epoch in range(epochs):
        for i, language in enumerate(languages):
            folder_path = os.path.join(data_path, language)
            for file_name in os.listdir(folder_path):
                with open(os.path.join(folder_path, file_name), 'r', encoding='ascii', errors='ignore') as file:
                    text_vector = preprocess_text(file.read())
                    outputs = weights.dot(text_vector)
                    target = np.zeros(num_languages)
                    target[i] = 1
                    error = target - outputs
                    weights += learning_rate * np.outer(error, text_vector)

                    weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)

    return weights, languages

def predict_language(text, weights, languages):
    vector = preprocess_text(text)
    outputs = weights.dot(vector)
    return languages[np.argmax(outputs)]


data_path = 'path_to_your_language_folders'
weights, languages = train_network(data_path)
input_text = "This is a test text."
