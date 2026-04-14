import json
import random
import os
import numpy as np

# nltk is lightweight — fine to import at module level
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


# ── Lazy chatbot model ────────────────────────────────────────────────────────
# torch (~1.5 GB) is only imported when the chatbot is first used.

_chatbot_model = None
_all_words = None
_tags = None
_intents = None


def _load_chatbot():
    global _chatbot_model, _all_words, _tags, _intents

    import torch

    with open('chatbot/intents.json', 'r') as f:
        _intents = json.load(f)

    FILE = os.path.join(os.getcwd(), 'chatbot/NeuralNet.pth')
    data = torch.load(FILE, map_location=torch.device('cpu'))

    input_size   = data["input_size"]
    hidden_size  = data["hidden_size"]
    output_size  = data["output_size"]
    _all_words   = data['all_words']
    _tags        = data['tags']

    class NeuralNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.l1   = torch.nn.Linear(input_size, hidden_size)
            self.l2   = torch.nn.Linear(hidden_size, hidden_size)
            self.l3   = torch.nn.Linear(hidden_size, output_size)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.l3(self.relu(self.l2(self.relu(self.l1(x)))))

    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(data["model_state"])
    model.eval()
    _chatbot_model = (model, torch)


def get_response(msg):
    global _chatbot_model, _all_words, _tags, _intents

    if _chatbot_model is None:
        _load_chatbot()

    model, torch = _chatbot_model

    sentence = tokenize(msg)
    X = bag_of_words(sentence, _all_words)
    X = torch.from_numpy(X.reshape(1, -1))

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = _tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob  = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in _intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I'm sorry, I didn't understand that. Could you clarify your question?"
