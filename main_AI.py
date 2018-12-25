import numpy as np


def makeWeights(brain):
    weights=[]
    for p,l in enumerate(brain[1:]):
        
        weight=2*np.random.random((brain[p],l)) - 1
        
        weights.append(weight)
    return weights

def transfer(x,deriv=False):  ### TANH
    if(deriv==True):
        return 1/(x*x+1)
    return np.tanh(x)

    

np.random.seed(1)
class Network:### ADD anti-learning
    def __init__(self,layers):
        self.weights=makeWeights(layers)
        self.motovation=makeWeights(layers)
        
    def forward(self,inputs):
        inputs=np.array([inputs])
        self.nodes=[inputs]
        for weight in self.weights:
            node=transfer(np.dot(self.nodes[-1],weight))
            self.nodes.append(node)
        return self.nodes[-1][0]
    
    def backward(self,output,learningRate=0.1): #### .T means the transpose
        output=np.array([output])
        error=output-self.nodes[-1]## nodes[-1]  is the output
        Ne=np.mean(np.abs(error))  ## Network Error
        delta=error*transfer(self.nodes[-1],True)
        deltas=[delta]
        for p, i in enumerate(list(reversed(self.nodes))[1:]):
            error=deltas[p].dot(self.weights[-(p+1)].T)###[-(p+1)] means the opposit end
            delta=error*transfer(i,True)
            deltas.append(delta)
            
        deltas=list(reversed(deltas))
        
        for p,i in enumerate(self.nodes[:-1]):
            z=i.T.dot(deltas[p+1])  
            self.weights[p] += z*learningRate
            ## EXPERIMENTAL MOTOVATION
            #self.weights[p] += self.motovation[p]*0.05
            #self.motovation[p]=z
        return Ne

    def save(self,file):
        np.save(file,np.array(self.weights))
    def load(self,file):
        self.weights=np.load(file)
