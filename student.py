#!/usr/bin/env python3

# used venv
# pip3 install torch==1.2.0 torchtext==0.4.0

"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import re
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

# str -> [str]
def tokenise(sample):
  """
  Called before any processing of the text has occurred.
  """
  processed = sample.split()
  return processed

# [str] ->
def preprocessing(sample):
  """
  Called after tokenising but before numericalising.
  """
  preprocessed = []
  for word in sample:
    clean_word = word.strip().lower()
    clean_word = re.sub('[^a-zA-Z]', '', clean_word)
    if len(clean_word) > 2:
      preprocessed.append(clean_word)
  return preprocessed


def postprocessing(batch, vocab):
  """
  Called after numericalising but before vectorising.
  """
  return batch

stopWords = ['i', 'ive', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

VectorSize = 50
wordVectors = GloVe(name='6B', dim=VectorSize)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
  """
  Your model will be assessed on the predictions it makes, which must be in
  the same format as the dataset ratings and business categories.  The
  predictions must be of type LongTensor, taking the values 0 or 1 for the
  rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
  outputs a different representation convert the output here.
  """
  ratingOutput = torch.round(tnn.Sigmoid()(ratingOutput)).long()
  categoryOutput = torch.argmax(categoryOutput, axis=1)
  # OLD
  # ratingOutput = tnn.Sigmoid()(ratingOutput).long()
  # categoryOutput = np.argmax(categoryOutput, axis=1)  # Won't work with cuda:0
  return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
  """
  Class for creating the neural network.  The input to your network will be a
  batch of reviews (in word vector form).  As reviews will have different
  numbers of words in them, padding has been added to the end of the reviews
  so we can form a batch of reviews of equal length.  Your forward method
  should return an output for both the rating angerd the business category.
  """

  # The rating network should only have 1 output node
  # The output node with the biggest value should be the predicted category
  # Create 2 models in one class and treat them individually to do different tasks
  def __init__(self, hiddenSize, numLayers, vectorSize):
    super(network, self).__init__()
    self.relu = tnn.ReLU()

    self.pDropOut = 0.3
    self.dropout = tnn.Dropout(self.pDropOut)

    self.hiddenSize = hiddenSize
    self.numLayers = numLayers
    self.vectorSize = vectorSize

    # Rating network
    self.initRatingModel()
    # Category network
    self.initCategoryModel()

  def initRatingModel(self):
    self.ratingLSTM = tnn.LSTM(input_size=self.vectorSize,
      hidden_size=self.hiddenSize,
      batch_first=True,
      num_layers=self.numLayers,
      bias=True,
      dropout=self.pDropOut,
      bidirectional=True
      )
    self.ratingLin1 = tnn.Linear(self.hiddenSize*self.numLayers, 64)
    self.ratingLin2 = tnn.Linear(64, 1)

  def initCategoryModel(self):
    self.categoryLSTM = tnn.LSTM(input_size=self.vectorSize,
      hidden_size=self.hiddenSize,
      batch_first=True,
      num_layers=self.numLayers,
      bias=True,
      dropout=self.pDropOut,
      bidirectional=True)
    self.categoryLin1 = tnn.Linear(self.hiddenSize*self.numLayers, 64)
    self.categoryLin2 = tnn.Linear(64, 5)

  # (nLabels, batchSize, hiddenSize)
  def forwardRating(self, input, length):
    # LSTM
    output, _ = self.ratingLSTM(input)

    # Linear
    output = self.ratingLin1(output[:, -1, :])
    output = self.relu(output)
    output = self.ratingLin2(output)
    return output.squeeze()

  def forwardCategory(self, input, length):
    # LSTM
    output, _ = self.categoryLSTM(input)

    # Linear
    output = self.categoryLin1(output[:, -1, :])
    output = self.relu(output)
    output = self.categoryLin2(output)
    return output.squeeze()

  # (batch_size, max(length), vector_size), tensor(32)
  # output -> [batch_size, 1], (batch_size, 5]
  def forward(self, input, length):
    return self.forwardRating(input, length), self.forwardCategory(input, length)

class loss(tnn.Module):
  """
  Class for creating the loss function.  The labels and outputs from your
  network will be passed to the forward method during training.
  """

  def __init__(self):
    super(loss, self).__init__()

  def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
    ratingOutput = ratingOutput.float()
    ratingTarget = ratingTarget.float()
    ratingLoss = tnn.BCEWithLogitsLoss()(ratingOutput, ratingTarget)
    categoryLoss = tnn.CrossEntropyLoss()(categoryOutput, categoryTarget)
    # loss = ratingLoss + categoryLoss
    return ratingLoss + categoryLoss

hiddenSize = 256      # number of hidden neurons
numLayers = 2         # number of layers
net = network(hiddenSize, numLayers, VectorSize)
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 96 # 32
epochs = 10
lr = 0.001
# optimiser = toptim.SGD(net.parameters(), lr=lr)
optimiser = toptim.Adam(net.parameters(), lr=lr)
