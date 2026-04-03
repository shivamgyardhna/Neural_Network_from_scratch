import numpy as np
class Layer_dropout:
  def __init__(self,rate):
    # Store rate, we invert it as for example for dropout
    # of 0.1 we need success rate of 0.9
    self.rate=1-rate
  def forward(self,input):
    self.inputs=input
    self.binary_mask= np.random.binomial(1,self.rate,input.shape)/(1-self.rate)
    self.output=self.inputs * np.random.binomial(1,self.rate,input.shape)
  def backward(self,dvalue):
    self.dinputs= dvalue *self.binary_mask