from sklearn.neural_network import MLPClassifier   
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 




class MultiLayerPerceptronClassifier(MLPClassifier, TfidfVectorizer):
	"""docstring for MultiLayerPerceptronClassifier"""
	def __init__(self, hiddien_layer = 80, neuron=80):
		self.feature_extraction = TfidfVectorizer()   
		self.clf = MLPClassifier(hidden_layer_sizes=(neuron,hiddien_layer),activation='relu', solver='adam', random_state=42)

	def fit(self, X, y, **kwargs):
		self.tfidf = self.feature_extraction.fit(X)
		X_train = self.tfidf.transform(X)
		self.clf.fit(X_train, y)
		return self


	def predict(self, X):
		print(dir(self))
		X_train = self.tfidf.transform([X])
		return self.clf.predict(X_train)

		
	def fit_predict(self, X, y, **kwargs):
		return self.fit(X, y).predict(X)
