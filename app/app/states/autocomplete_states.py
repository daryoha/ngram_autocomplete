import reflex as rx

from typing import List
from collections import deque
from typing import Set, Dict
import pandas as pd


class PrefixTreeNode:
    def __init__(self):
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False

    def get_descendants(self, prefix) -> List[str]:
        results: List[str] = []

        def dfs(current, path):
            if current.is_end_of_word:
                results.append(prefix + "".join(path))
            for ch, child in current.children.items():
                dfs(child, path + [ch])

        dfs(self, [])
        return results

class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()

        for word in vocabulary:
            self._insert(word)

    def _insert(self, word: str) -> None:
        node = self.root
        for c in word:
            node = node.children.setdefault(c, PrefixTreeNode())
        node.is_end_of_word = True

    def search_prefix(self, prefix) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """
        
        node = self.root
        
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        return node.get_descendants(prefix)


from collections import Counter
from nltk.corpus import wordnet
import numpy as np
import pickle


class WordCompletor:
    def __init__(self, corpus, min_freq = 0):
        """
        corpus: list – корпус текстов
        """
        flat_words = (word for doc in corpus for word in doc)
        freq = Counter(flat_words)
        freq = Counter({w: c for w, c in freq.items() if c >= min_freq})

        self.sum_freqs = sum(freq.values())
        self.voc = freq
        self.prefix_tree = PrefixTree(self.voc.keys())

    def get_words_and_probs(self, prefix: str):
        """
        Возвращает список слов, начинающихся на prefix,
        с их вероятностями (нормировать ничего не нужно)
        """
        words, probs = [], []
        words = self.prefix_tree.search_prefix(prefix)
        for word in words:
            probs.append(self.voc[word])
        probs = np.array(probs)
        probs = list(probs / self.sum_freqs)
        return words, probs
    

from collections import defaultdict
import re
import string
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import ast
import os

class NGramLanguageModel:
    def __init__(self, corpus, n, corpus_name):
        self.n = n
        self.prefix_counters = defaultdict(Counter)
        if os.path.exists(f"prefix_counters_{n}_{corpus_name}.pkl"):
            with open(f"prefix_counters_{n}_{corpus_name}.pkl", "rb") as f:
                self.prefix_counters = pickle.load(f)
        else:
            for text in corpus:
                if isinstance(text, str):
                    text = ast.literal_eval(text) 
                padded = ['<s>'] * (n-1) + text + ['<e>']
                for i in range(len(text)):
                    prefix = tuple(padded[i:i+n])
                    next_word = padded[i+n]
                    self.prefix_counters[prefix][next_word] += 1

            with open(f"prefix_counters_{n}_{corpus_name}.pkl", "wb") as f:
                pickle.dump(self.prefix_counters, f)
                
        self.prefix_totals = {
            prefix: sum(counter.values())
            for prefix, counter in self.prefix_counters.items()
        }

        

    def get_next_words_and_probs(self, prefix: list) -> (List[str], List[float]):
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов
        """

        next_words, probs = [], []
        prefix = prefix[-self.n:]
        key = tuple(prefix)
        if key not in self.prefix_counters:
            return [], []
        counter = self.prefix_counters[key] 
        next_words = counter.keys()
        total = self.prefix_totals[key]

        probs = [count / total for count in counter.values()]

        return next_words, probs
    
import heapq
from copy import deepcopy
import math

class TextSuggestion:
    def __init__(self, word_completor, n_gram_model):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model

    def beam_search_extend(self, text, n_gram_model, n_words, beam_width):
        beam = [(0.0, text)]
        
        for _ in range(n_words):
            candidates = []   
            for logp, seq in beam:
                next_words, probs = n_gram_model.get_next_words_and_probs(seq)
                for word, p in zip(next_words, probs):
                    if p < 1e-2:
                        continue
                    if word == '<e>':
                        continue
                    new_seq = seq + [word]
                    new_logp = logp + math.log(p)
                    candidates.append((new_logp, new_seq))
            
            beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
        
        return beam

    def suggest_text(self, text: list, n_words=3, n_texts=1, method='beam') -> list[list[str]]:
        """
        Возвращает возможные варианты продолжения текста (по умолчанию только один)
        
        text: список слов – написанный пользователем текст
        n_words: число слов, которые дописывает n-граммная модель
        n_texts: число возвращаемых продолжений (пока что только одно)
        
        return: list[list[srt]] – список из n_texts списков слов, по 1 + n_words слов в каждом
        Первое слово – это то, которое WordCompletor дополнил до целого.
        """

        words, probs = word_completor.get_words_and_probs(text[-1])
        if len(words) == 0:
            return []
        text[-1] = words[np.array(probs).argmax()]

        if method == 'beam':
            texts_n_probs = self.beam_search_extend(text, self.n_gram_model, n_words, n_texts)
            if len(texts_n_probs):
                return [text[-n_words-1:] for prob, text in texts_n_probs]
            else:
                return [[text[-1]]]
        



def clean_and_tokenize(text: str) -> list:
    parts = re.split(r'\r?\n\r?\n', text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]
    
    body = re.sub(r'http[s]?://\S+', ' ', body)
    body = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b', ' ', body)
    
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    no_punct = body.translate(translator)
    
    tokens = word_tokenize(no_punct, language='english')
    tokens = [tok.lower() for tok in tokens if tok.isalpha()]
    
    return tokens

import json

print("make corpus")
if not os.path.exists("corpus.json"):
    emails = pd.read_csv('emails.csv')
    emails['cleaned_message'] = emails['message'].apply(clean_and_tokenize)
    corpus = list(emails['cleaned_message'].values)
    with open("corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

print("read corpus")
with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

print("word")
word_completor = WordCompletor(corpus)
print("ngram")
n_gram_model = NGramLanguageModel(corpus=corpus, n=2, corpus_name="emails")
print("text")
text_suggestion = TextSuggestion(word_completor, n_gram_model)
print("ready")


class AutocompleteState(rx.State):
    """Manages the state for the autocomplete component."""
    input_text: str = ""
    show_suggestions: bool = False
    
    

    @rx.var
    def filtered_items(self) -> list[str]:
        """Filters items based on the input text."""

        if self.input_text == "":
            return []

        print("start suggestion")
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        no_punct = self.input_text.translate(translator)
        tokens = word_tokenize(no_punct, language='english')
        tokens = [tok.lower() for tok in tokens if tok.isalpha()]
        suggetstions = [' '.join(tokens[:-1]) + (' ' if tokens[:-1] else '') + ' '.join(text) for text in text_suggestion.suggest_text(['<s>'] * (n_gram_model.n-1) + tokens, n_words=3, n_texts=5)]
        print(suggetstions)
        return suggetstions
        

    def set_input_text(self, text: str):
        """Sets the input text and shows suggestions."""
        self.input_text = text
        self.show_suggestions = True

    def select_item(self, item: str):
        """Selects an item from the suggestions, updating the input field."""
        self.input_text = item
        self.show_suggestions = False

    def hide_suggestions(self):
        """Hides the suggestions dropdown."""
        print("hide")
        self.show_suggestions = False