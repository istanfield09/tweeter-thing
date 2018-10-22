import sys
import random

import numpy
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

import fake_corpus

CHUNK_LEN = 40
STEP = 3
DIVERSITIES = [0.2, 0.7, 1.2]

class TweetGenerator(object):
        """
        An object used to derive similar sounding tweets from a tweet corpus.
        Each iteration strengthens character predictions, resulting in better
        tweets.
        
        Uses snippets from Natural Languages Processing with 
        Python (Krishna Bhavsar, Naresh Kumar, Pratap Dangeti).
        """
	def __init__(self):
		self.characters = []
		self.character_to_index = {}
		self.index_to_character = {}
		self.tweet_corpus = []

		self.chunks = []
		self.next_char_from_chunk = []

		self.model = Sequential()

		self._populate_corpus()
		self._build_mappings()

	def _populate_corpus(self):
		tweet_data = fake_corpus.get_tweet_corpus()
		"""
		We create a corpus of all tweets, duplicated so that we can 
		arbritrarily sample the tweets on a per character basis across a
		reasonably large dataset.
		"""
		tweets = [
			tweet.get("text", "") for tweet in tweet_data
		] * 8
		random.shuffle(tweets)
		self.tweet_corpus = tweets

	def _build_mappings(self):
                """
                Constructs a map in both directions of characters to indices
                to characters. Also derives a sampling of characters and characters
                that follow them to build a prediction model with.
                """
		joined_corpus = " ".join(self.tweet_corpus)
		for i in xrange(0, len(joined_corpus) - CHUNK_LEN, STEP):
			self.chunks.append(joined_corpus[i:i + CHUNK_LEN])
			self.next_char_from_chunk.append(joined_corpus[i + CHUNK_LEN])

		self.characters = sorted(list(set(joined_corpus)))
		for index, character in enumerate(self.characters):
			self.character_to_index[character] = index
			self.index_to_character[index] = character
		self.corpus = joined_corpus

	def _vectorize_indices_preprocess(self):
                """
                Builds vectors used to predict next character.
                """
		x = numpy.zeros(
			(len(self.chunks), CHUNK_LEN, len(self.characters)),
			dtype=numpy.bool
		)
		y = numpy.zeros(
			(len(self.chunks), len(self.characters)),
			dtype=numpy.bool
		)

		for i, chunk in enumerate(self.chunks):
			for j, char in enumerate(chunk):
				x[i, j, self.character_to_index[char]] = 1
				y[i, self.character_to_index[self.next_char_from_chunk[i]]] = 1
		return x, y

	def _construct_model(self):
		self.model.add(
			LSTM(128, input_shape=(CHUNK_LEN, len(self.characters)))
		)
		self.model.add(Dense(len(self.characters)))
		self.model.add(Activation('softmax'))
		self.model.compile(
			loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01)
		)

	def _predicted_indices(self, predictions, metric=1.0):
		predictions = numpy.asarray(predictions).astype('float64')
		predictions = numpy.log(predictions) / metric
		exp_predictions = numpy.exp(predictions)
		predictions = exp_predictions / numpy.sum(exp_predictions)
		probabilities = numpy.random.multinomial(1, predictions, 1)
		return numpy.argmax(probabilities)

	def learn(self):
		print "CHARACTERS: ", len(self.characters)
		print "CHUNKS: ", len(self.chunks)
		print "CORPUS LENGTH: ", len(self.corpus)
		vectorized_x, y = self._vectorize_indices_preprocess()
		self._construct_model()
		for iteration in range(1, 30):
			print "-" * 40
			print "iteration: ", iteration
			self.model.fit(vectorized_x, y, batch_size=128, epochs=1)
			start_index = random.randint(
				0, len(self.corpus) - CHUNK_LEN - 1
			)

			for diversity in DIVERSITIES:
				print "diversity, ", diversity
				generated = ''
				sentence = self.corpus[start_index:start_index + CHUNK_LEN]
				print "USING SENTENCE: ", sentence
				generated += sentence
				for i in range(240):
					x = numpy.zeros((1, CHUNK_LEN, len(self.characters)))
					for j, char in enumerate(sentence):
						x[0, j, self.character_to_index[char]] = 1.

					preds = self.model.predict(x, verbose=0)[0]
					next_index = self._predicted_indices(preds, diversity)
					predicted_character = self.index_to_character[next_index]
					sys.stdout.write(predicted_character)
					generated += predicted_character
					sentence = sentence[1:] + predicted_character
				sys.stdout.write("\n")




if __name__ == "__main__":
	tg = TweetGenerator()
	tg.learn()
