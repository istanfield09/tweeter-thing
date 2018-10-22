import json
import os

import pymongo


MONGO_HOST = os.environ.get('MONGO_HOST')
MONGO_PORT = os.environ.get('MONGO_PORT')
MONGO_CLIENT = pymongo.MongoClient(MONGO_HOST, int(MONGO_PORT))
TARGET_DB = 'tweets'


def get_data_from_json(json_file):
	json_file = os.path.abspath(json_file)
	with open(json_file) as open_file:
		data = json.load(open_file)

	return data


def write_data_to_db(target_collection, data):
	"""
	Takes a list of json documents and writes to target collection
	of TARGET_DB
	"""
	db = getattr(MONGO_CLIENT, TARGET_DB)
	collection = getattr(db, target_collection)

	return collection.insert_many(data)

def get_all_from_collection(target_collection):
	"""
	Retrieves all documents from a collection of TARGET_DB
	"""
	db = getattr(MONGO_CLIENT, TARGET_DB)
	collection = getattr(db, target_collection)
	return collection.find()

def get_tweet_corpus():
	"""
	Retrieves list of all tweets in `tweets_collection`
	"""
	return get_all_from_collection('tweets_collection')


if __name__ == "__main__":
	TWEET_FILE = 'tweets.json'
	TWEET_COLLECTION = 'tweets_collection'

	data = get_data_from_json(TWEET_FILE)
	tweet_data = data.get('tweets', [])
	tweet_data = [ {"text": tweet} for tweet in tweet_data ]
	write_data_to_db(TWEET_COLLECTION, tweet_data)

