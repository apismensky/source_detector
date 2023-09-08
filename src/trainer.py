#!/usr/bin/env python

import os
import shutil
import os
import io
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train():
    directory_path = os.path.join('../source')
    output_path = '../source'
    java_path = os.path.join(output_path, 'java')
    cpp_path = os.path.join(output_path, 'cpp')
    scala_path = os.path.join(output_path, 'scala')
    js_path = os.path.join(output_path, 'javascript')
    py_path = os.path.join(output_path, 'python')
    text_path = os.path.join(output_path, 'plaintext')

    data = DataFrame({'message': [], 'class': []})
    data = pd.concat([data, dataFrameFromDirectory(cpp_path, "cpp")])
    data = pd.concat([data, dataFrameFromDirectory(java_path, "java")])
    data = pd.concat([data, dataFrameFromDirectory(js_path, "javascript")])
    data = pd.concat([data, dataFrameFromDirectory(py_path, "python")])
    data = pd.concat([data, dataFrameFromDirectory(scala_path, "scala")])
    data = pd.concat([data, dataFrameFromDirectory(text_path, "text")])

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer(tokenizer=custom_tokenizer)
    counts = vectorizer.fit_transform(train_data['message'].values)

    classifier = MultinomialNB()
    targets = train_data['class'].values
    classifier.fit(counts, targets)

    # Check the accuracy of the model
    test_messages = vectorizer.transform(test_data['message'].values)
    predictions = classifier.predict(test_messages)
    true_labels = test_data['class'].values
    accuracy = accuracy_score(true_labels, predictions)
    print("Accuracy:", accuracy)

    # Calculate precision, recall, and F1-score
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    # Create a confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:\n", conf_matrix)

    return vectorizer, classifier

def remove_comments(code):
    # Remove single-line comments starting with "//" or "#"
    code = re.sub(r'(\/\/[^\n]*|#[^\n]*)', '', code)

    # Remove multi-line comments enclosed within '/*' and '*/'
    code = re.sub(r'\/\*[\s\S]*?\*\/', '', code)

    return code

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            # remove comments
            # todo in the real world we may need to add more robust logic for tokenization
            # ie for prgramming languages we want to exclude string and numeric literals, names (field, method, variables) etc and may be leave types, operands,  braces, brackets, spaces and reserved words?
            message = remove_comments(message)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

# Custom tokenizer that tokenizes based on braces, brackets, and spaces. Those are the most common programming characters, we want to count them as separate tokens.
programming_chars = ['(', ')','{', '}', '[', ']', ' ', '=', '+', '-', '*', '/', '#', '!', '^', '?', '"', "'", ';', '.']
def custom_tokenizer(text):
    tokens = []
    current_token = ""
    for char in text:
        if char in programming_chars:
            if current_token:
                tokens.append(current_token)
            tokens.append(char)
            current_token = ""
        else:
            current_token += char
    if current_token and len(current_token.strip()) > 0:
        tokens.append(current_token)
    return tokens
