import argparse


class Parser:

    def __init__(self):

        self._parser = argparse.ArgumentParser(
            description='Multilayered Perceptron')

        self._parser.add_argument('-d', '--dataset', nargs='?', default='mnist', choices=['tarasiuk', 'mnist', 'abalone'],
                                  help='Dataset name')

        self._parser.add_argument('-m', '--mode', default=0, type=int,
                                  help='0 for learn, 1 for test')

        self._parser.add_argument('-hid', '--hidden', nargs='+', type=int,
                                  help='Number of hidden layers and number of neurons per layer')

        self._parser.add_argument('-a', '--activation_functions', nargs='+', default=['sigmoid', 'sigmoid', 'sigmoid'], choices=['sigmoid', 'tanh', 'softsign', 'relu', 'leakyrelu'],
                                  help='Activation functions for every hidden layer + output layer')

        self._parser.add_argument('-e', '--epochs', type=int, default=10,
                                  help='Number of epochs')

        self._parser.add_argument('--lambda', type=float, default=0.005, dest='lamb',
                                  help='Set lambda')

        self._parser.add_argument('--momentum', type=float, default=0.2,
                                  help='Set momentum')

        self._parser.add_argument('--no-plot', type=int, default=0,
                                  help='Choose 1 for no error plot')

        self._parser.add_argument('-err', '--error-treshold', type=float, default=0,
                                  help='Error treshold, surpassing it will cause program to stop')

        self._parser.add_argument('-sr', '--serialize',
                                  help='File name of serialization')

    def parse(self):
        self._args = self._parser.parse_args()

    def get_dataset(self):
        return self._args.dataset

    def get_mode(self):
        return self._args.mode

    def get_hidden(self):
        if self._args.hidden is None:
            if self._args.dataset == 'abalone':
                return [2, 18, 18]
            else:
                return [2, 397, 397]

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

    def get_activation_functions(self):
        return self._args.activation_functions

    def get_path(self):
        if self._args.serialize is None:
            return self._args.dataset + '.p'
