import numpy as np
import random as rnd
import math 
from abc import ABCMeta, abstractmethod


#Comparing two lists
#<param name="list1">first list (float)</param>
#<param name="list2">Second list (float)</param>
#<returns>Comparing result (float)</returns>
def list_compare(list1, list2):
    if len(list1) != len(list2):
        s=str(len(list1))+" "+str(len(list2))
        raise Exception("Lists must be the same size "+s)
    if len(list1) == 1:
        return abs(float(list1[0]) - float(list2[0]))
    res = 0.0
    i=0
    while i<len(list1):
        dif = float(list1[i]) - float(list2[i])
        res=res+(dif*dif)
        i=i+1
    return math.sqrt(res)

#Abstract class transfer function (or Transmission function)
class TransFunc():
    __metaclass__=ABCMeta

    @abstractmethod
    def compute(self,income):
        """Calculate transfer function"""
    
    @abstractmethod
    def get_id(self):
        """Transfer function ID"""

    #level - уровнь мутации
    @abstractmethod
    def mutation(self, level):
        """Transfer function mutation"""

    @abstractmethod
    def clone(self):
        """Copy of the transfer function"""             

#Transfer function "As is"
class AsIs(TransFunc):

    def compute(self,income):
        return income

    def get_id(self):
        return 1

    def clone(self):
        return AsIs()

    def mutation(self, level):
        return        
       
assert issubclass(AsIs, TransFunc)
assert isinstance(AsIs(), TransFunc)


#Transfer function "Sigmoid"
class SignFunc(TransFunc):
    def __init__(self):
        self.par=1
        
    def compute(self,income):
        return 1/(1+np.exp(-self.par*income))

    def get_id(self):
        return 2
    
    def clone(self):
        trans=SignFunc()
        trans.par=self.par
        return trans

    def mutation(self, level):
        self.par = self.par + rnd.random() * level - level / 2.0 

    
assert issubclass(SignFunc, TransFunc)
assert isinstance(SignFunc(), TransFunc)

#Transfer function "Threshold"
class ThresholdFunc(TransFunc):
    def compute(self,income):
        if income>0:
            return 1
        else:
            return 0

    def get_id(self):
        return 3

    def clone(self):
        return ThresholdFunc()

    def mutation(self, level):  
        return
    
assert issubclass(ThresholdFunc, TransFunc)
assert isinstance(ThresholdFunc(), TransFunc)

#Neuron class
class Neuron:
    def __init__(self, count, trans):
        self.inputs=[0]*count 
        self.weights=[rnd.random()-0.5 for i in range(count+1)]
        self.trans=trans
        self.output=0

    #Calculate neuron
    def compute(self):
        res=0
        i=1
        count=len(self.weights)
        while i<count:
            res = res+(self.weights[i] * self.inputs[i-1])
            i=i+1
        res=res+self.weights[0]
        self.output = self.trans.compute(res)
        
    #Create a copy of a neuron
    def clone(self):
        res = Neuron(len(self.weights)-1,self.trans.clone())
        i=0
        count=len(self.weights)
        while i<count:
            res.weights[i] = self.weights[i]
            i=i+1
        return res

    def mutation(self,level):
        i=0
        count=len(self.weights)
        while i<count:
            self.weights[i] = self.weights[i] + rnd.random() * level - level / 2.0
            i=i+1
        
#Layer class
class Layer:
    #Constructor
    #count - number of neurons
    #inputs_count - number of inputs in each neuron
    #trans - Transmission function
    def __init__(self, count, inputs_count, trans=""):
        self.inputs_count=inputs_count
        self.outputs=[]
        if count>0 and trans!="":
            self.neurons=[Neuron(inputs_count, trans.clone())]*count
        else:
            self.neurons=[]

    #Duplicate layer
    def clone(self):
        i=0
        count=len(self.neurons)
        res = Layer(0,self.inputs_count)
        while i<count:
            res.neurons.append(self.neurons[i].clone())
            i=i+1
        res.count=count
        return res

    #Calculate layer
    def compute(self):
        i=0
        count=len(self.neurons)
        self.outputs=[]
        while i<count:
            self.neurons[i].compute()
            self.outputs.append(self.neurons[i].output)
            i=i+1

    #Set input parameters
    def set_incomes(self, inputs):
        count=len(self.neurons)
        count_inputs=self.inputs_count
        i=0
        while i<count:
            j=0
            while j<count_inputs:
                self.neurons[i].inputs[j]=inputs[j]
                j=j+1
            i=i+1

    def mutation(self,level):
        i=0
        count=len(self.neurons)
        while i<count:
            self.neurons[i].mutation(level)
            i=i+1            

#Neural Network class
class NeuralNet:
    #Constructor
    #input_count - number of inputs
    #output_count - number of outputs
    def __init__(self,input_count,output_count):
        self.target_function = False
        self.layers=[]
        if input_count>0:
            self.inputs=[0]*input_count
        else:
            self.inputs=[]
        if output_count>0:
            self.outputs=[0]*output_count
        else:
            self.outputs=[]

    #Set input parameters
    #<param name="a_incomes">inputs data</param>
    def set_incomes(self, a_incomes):
        self.inputs=a_incomes

    #Create a copy of the neural network
    def clone(self):
        i=0
        count=len(self.layers)
        res = NeuralNet(len(self.inputs), len(self.outputs))
        while i<count:
            res.layers.append(self.layers[i].clone())
            i=i+1
        return res

    #Calculate neural network
    def compute(self):
        i=0
        count=len(self.layers)
        self.layers[0].set_incomes(self.inputs)
        while i<count:
            self.layers[i].compute()
            if i<count-1:
                self.layers[i+1].set_incomes(self.layers[i].outputs)
            i=i+1

        i=0
        count_outputs=len(self.layers[count-1].outputs)
        while i<count_outputs:
            self.outputs[i]=self.layers[count-1].outputs[i]
            i=i+1

    def mutation(self,level):
        i=0
        count=len(self.layers)
        while i<count:
            self.layers[i].mutation(level)
            i=i+1

    #Get layer states
    def get_layers_conf(self):
        i=0
        res=[]
        count=len(self.layers)
        while i<count:
            res.append(len(self.layers[i].neurons))
            i=i+1
        return res

    #/// Create layer by sigmoidal function parameter value
    #/// <param name="count">number of neurons</param>
    #/// <param name="inputs_count">number of inputs</param>
    #/// <param name="param">sigmoidal function parameter value</param>
    def create_layer(self,count, inputs_count, param):
        sg=SignFunc()
        sg.par=param
        layer = Layer(count, inputs_count, sg)
        self.layers.append(layer)

    #print neural network
    def print(self):
        print("------")
        count=len(self.layers)
        i=0
        while i<count:
            ncount=len(self.layers[i].neurons)
            print("Layer ",i," of neurons: ",)
            j=0
            while j<ncount:
                neuron=self.layers[i].neurons[j]
                print("  Neurons ",j," inputs ",len(neuron.inputs)," weights ",len(neuron.weights)," trans ", neuron.trans.get_id())
                icount=len(neuron.inputs)
                k=0
                while k<icount:
                    print("    inputs ",k,"=",neuron.inputs[k])
                    k=k+1
                j=j+1
            i=i+1    
        print("------")

#Training sample class
class StudyMatrixItem:

    #Constructor
    #a_incomes - inputs
    #a_outcomes - outputs
    def __init__(self,a_incomes, a_outcomes):
        self.incomes=a_incomes #inputs
        self.outcomes=a_outcomes #outputs
            
#Genetic Algorithm Class
class GeneticAlgorith:

    #Constructor
    def __init__(self):
    
        #The minimum number of individuals in the population (if it turns out to be less than this number, then we do not make selection)
        self.min_count = 5

        #The maximum number of individuals in the population (if it turns out to be more than this number, then we cut it to this number)
        self.max_count = 30

        #Initial number of places in the population
        self.count=10

        #Allow Transfer Function Replacement Mutation
        self.allow_change_trans_function=True

        self.p_mutation=0.1 #Mutation probability
        self.population=[] #population
        self.selection=[] #Training sample
        self.testing=[] #Test sample

        self.level=0.5 #mutation level

    #Initialize from neural network
    #net - neural network
    #count_source_mutation - The number of mutations in the initial neural network in the initial sample
    def init_population_from_net(self, net, count_source_mutation):        
        count_mut = count_source_mutation
        i=1
        while i<=self.count:
            lnet = None
            if count_mut>0:
                if count_mut == count_source_mutation:
                    lnet = net
                else:
                    lnet = net.clone()
                    lnet.mutation(self.level)
                count_mut=count_mut-1
            else:
                lnet = NeuralNet(len(net.inputs), len(net.outputs))
                size = len(net.inputs)
                conf = net.get_layers_conf()
                for item in conf:
                    lnet.create_layer(item, size, 1)
                    size = item
            self.population.append(lnet)
            i=i+1
        self.sorting()


    #/// reproduction
    def reproduction(self):
        self.count=len(self.population)
        i=0
        while i<self.count:
            j=i+1
            while j<self.count:
                p=self.cross_probability(i,j,self.count)
                if rnd.random()<p:        
                    self.population.append(self.cross(self.population[i],self.population[j]))
                j=j+1
            i=i+1

    #/// Calculate the probability of crossing
    #/// <param name="net1">First neural network (number)</param>
    #/// <param name="net2">Second neural network (number)</param>
    #/// <param name="count">The number of "individuals" in the population</param>
    #/// <returns>Crossing probability</returns>
    def cross_probability(self,net1, net2, count):
        return (2.0*float(count)-float(net1)-float(net2)) / (2.0*float(count)-1.0)
            
    #/// Crossing neural networks
    #/// <param name="net1">First neural network</param>
    #/// <param name="net2">Second neural network</param>
    #/// <returns>Crossing result</returns>
    def cross(self,net1, net2):
        res = NeuralNet(len(net1.inputs), len(net1.outputs))
        i=0
        while i<len(net1.layers):
            layer1=net1.layers[i]
            layer2=net2.layers[i]
            l_layer = Layer(count=0,inputs_count=layer1.inputs_count)
            j=0
            count = rnd.randint(0,len(layer1.neurons)-1)  
            while j<=count:
                l_layer.neurons.append(layer1.neurons[j].clone())
                j=j+1
            j = count + 1
            while j<len(layer1.neurons):
                l_layer.neurons.append(layer2.neurons[j].clone())
                j=j+1
            res.layers.append(l_layer)
            i=i+1
        if rnd.random()<self.p_mutation:
            res.mutation(self.level)
        return res
      

    #Carry out selection
    def selecticting(self):
        if len(self.population)<self.min_count:
            return
        items_for_removed=[]

        #We start with the second element (index 1), since the first (index 0) should survive anyway)
        while len(self.population)>self.max_count:
            i = 1
            _count=float(len(self.population))
            while i < len(self.population):
                if rnd.random()<float(i)/_count:
                    items_for_removed.append(self.population[i])
                i=i+1

            #delete selected for deletion
            for item in items_for_removed:
                self.population.remove(item)

            #clear the list to not to delete it a second time
            items_for_removed.clear()

    #Sort by transfer function
    @staticmethod
    def sort_by_target_function(net):
        if net.target_function==False:
            return 99999999999999
        else:
            return net.target_function

    #Set input
    #<param name="net">Neural network (GANeuralNet)</param>
    #<param name="item">Inputs data (StudyMatrixItem)</param>
    def set_incomes(self, net, item):
        net.set_incomes(item.incomes)

    #Calculate the objective function  
    def calk_target_function(self,net):
        Esumm = 0.0
        i = 0
        count=float(len(self.selection))
        while i < len(self.selection):
            item = self.selection[i]
            res = item.outcomes
            self.set_incomes(net,item)
            net.compute()
            Esumm = Esumm+list_compare(res, net.outputs)
            i=i+1
        net.target_function = Esumm/count
              
    #Population sorting
    def sorting(self):
                    
        #first we need to calculate the objective function
        i = 1
        while i < len(self.population):
            self.calk_target_function(self.population[i])
            i=i+1

        #sorting
        self.population.sort(key=self.sort_by_target_function)

        self.best_net = self.population[0]                  

    #next era
    def next_age(self):
        self.prev_spec = self.population[0]
        self.reproduction()
        self.sorting()
        self.selecticting()