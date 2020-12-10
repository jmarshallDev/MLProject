# Preprocessing all.txt for use in algorithms

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import word_tokenize, WordNetLemmatizer, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import re
import pdb

ROOT = os.path.dirname(os.path.abspath(__file__))

def gimme():
	stuff_n_things = np.loadtxt(os.path.join(ROOT, 'all.txt'), delimiter='\t', dtype=str)

	# Separate Data and Labels
	data = stuff_n_things[:,0].astype(str)
	labels = stuff_n_things[:,1].astype(int)

	# Apply String Manipulations
	data = string_fix(data)

	# Tokenize & Lemmetize Sentence
	data, empty_indices = process(data.astype(object))

	# Remove Empty Arrays
	data = np.delete(data, empty_indices)

	# Calculate TF-IDF Score
	tfidf_vectorizer = TfidfVectorizer(lowercase=False)

	to_sentence(data)

	data = tfidf_vectorizer.fit_transform(data)

	data = data.todense()

	return data, labels


def string_fix(arr):
	"""Converts Sentences to Lowercase and Removes Non-Letter Characters"""
	# Converting to Lowercase
	arr = np.char.lower(arr)

	for i in range(arr.size):
		# Replacing Non-Letter Characters with Spaces
		arr[i] = re.sub('[^a-z]', " ", arr[i])

	return arr


def process(arr):
	"""Removes Stop Words, Lemmatizes Sentences, and Notes Empty Arrays"""
	lemmatizer = WordNetLemmatizer()
	empty_indices = []

	for i in range(arr.size):
		# Removing Stop Words
		stop_words = set(stopwords.words("english"))
		words = word_tokenize(arr[i])
		without_stops = [word for word in words if not word in stop_words]

		# Lemmatizing
		sentence = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in without_stops]

		arr[i] = np.array(sentence)

		# Noting Empty Arrays for Later Removal
		if arr[i].size == 0:
			empty_indices.append(i)

	return arr, empty_indices


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def to_sentence(arr):
	"""Converts np array of Words into a String"""
	for i in range(arr.size):
		arr[i] = ' '.join(str(w) for w in arr[i])
