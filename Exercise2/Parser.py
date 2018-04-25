import argparse

class Parser:
    
	def __init__(self):

		self._parser = argparse.ArgumentParser(description='Multilayered Perceptron')

		self._parser.add_argument('-d', '--dataset', nargs = '?', default = 'abalone', choices = ['matrix', 'mnist', 'abalone'],
							help = 'Dataset name')

		self._parser.add_argument('-m','--mode', default = 2, type = int,
                            help = '0 for learn, 1 for neural-stuff, 2 for both')

		self._parser.add_argument('-hid','--hidden', nargs = '+', type = int, default = [1, 4],
                            help = 'Number of hidden layers and number of neurons per layer')

		self._parser.add_argument('-e','--epochs', type = int, default = 1000,
                            help = 'Number of epochs')

		self._parser.add_argument('--lambda', type = float, default = 0.9, dest = 'lamb',
                            help = 'Set lambda')

		self._parser.add_argument('--momentum', type = float, default = 0.6,
                            help = 'Set momentum')

		self._parser.add_argument('--no-plot', type = int, default = 0,
                            help = 'No error plot')

		self._parser.add_argument('-err', '--error-treshold', type = float, default = 0,
                            help = 'Error treshold, surpassing it will cause program to stop')

	def parse(self):
		self._args = self._parser.parse_args()

	def get_dataset(self):
		return self._args.dataset

	def get_mode(self):
		return self._args.mode

	def get_hidden(self):
		return self._args.hidden

	def get_epochs(self):
		return self._args.epochs

	def get_lambda(self):
		return self._args.lamb

	def get_momentum(self):
		return self._args.momentum

	def get_error(self):
		return self._args.error_treshold

	def get_plot(self):
		return self._args.no_plot
