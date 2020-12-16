import numpy as np
import random as rnd
import math 
from abc import ABCMeta, abstractmethod


#Сравнение двух списков
#<param name="list1">Первый список (список float)</param>
#<param name="list2">Второй список (список float)</param>
#<returns>Результат сравнения (float)</returns>
def list_compare(list1, list2):
    if len(list1) != len(list2):
        s=str(len(list1))+" "+str(len(list2))
        raise Exception("Списки должны иметь одинаковый размер "+s)
    if len(list1) == 1:
        return abs(float(list1[0]) - float(list2[0]))
    res = 0.0
    i=0
    while i<len(list1):
        dif = float(list1[i]) - float(list2[i])
        res=res+(dif*dif)
        i=i+1
    return math.sqrt(res)

#Абстрактный класс передаточной функции
class TransFunc():
    __metaclass__=ABCMeta

    @abstractmethod
    def compute(self,income):
        """Рассчитать передаточную функцию"""
    
    @abstractmethod
    def get_id(self):
        """ИД передаточной функции"""

    #level - уровнь мутации
    @abstractmethod
    def mutation(self, level):
        """Мутация передаточной функции"""

    @abstractmethod
    def clone(self):
        """Копия передаточной функции"""             

#Передаточная функция "Как есть"
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


#Передаточная функция "Сигмоида"
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

#Передаточная функция "Пороговая"
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

#Класс нейрона
class Neuron:
    def __init__(self, count, trans):
        self.inputs=[0]*count 
        self.weights=[rnd.random()-0.5 for i in range(count+1)]
        self.trans=trans
        self.output=0

    #Рассчитать нейрон
    def compute(self):
        res=0
        i=1
        count=len(self.weights)
        while i<count:
            res = res+(self.weights[i] * self.inputs[i-1])
            i=i+1
        res=res+self.weights[0]
        self.output = self.trans.compute(res)
        
    #Создать копию нейрона
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
        
#Класс слоя
class Layer:
    #Конструктор
    #count - количество нейронов
    #inputs_count - количество входов в каждом нейроне
    #trans - передаточная функция
    def __init__(self, count, inputs_count, trans=""):
        self.inputs_count=inputs_count
        self.outputs=[]
        if count>0 and trans!="":
            self.neurons=[Neuron(inputs_count, trans.clone())]*count
        else:
            self.neurons=[]

    #Создать копию слоя
    def clone(self):
        i=0
        count=len(self.neurons)
        res = Layer(0,self.inputs_count)
        while i<count:
            res.neurons.append(self.neurons[i].clone())
            i=i+1
        res.count=count
        return res

    #Рассчитать слой
    def compute(self):
        #res=0
        i=0
        count=len(self.neurons)
        self.outputs=[]
        while i<count:
            self.neurons[i].compute()
            self.outputs.append(self.neurons[i].output)
            i=i+1

    #Установить входные параметры
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

#Класс нейросети
class NeuralNet:
    #Конструктор
    #input_count - количество входов
    #output_count - количество выходов
    def __init__(self,input_count,output_count):
        #self.target_function="Не расчитана"
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

    #Установить входные параметры
    #<param name="a_incomes">Входные данные</param>
    def set_incomes(self, a_incomes):
        self.inputs=a_incomes

    #Создать копию нейросети
    def clone(self):
        i=0
        count=len(self.layers)
        res = NeuralNet(len(self.inputs), len(self.outputs))
        while i<count:
            res.layers.append(self.layers[i].clone())
            i=i+1
        return res

    #Рассчитать нейросеть
    def compute(self):
        #res=0
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

    #Получить конфигурацию слоев
    def get_layers_conf(self):
        i=0
        res=[]
        count=len(self.layers)
        while i<count:
            res.append(len(self.layers[i].neurons))
            i=i+1
        return res

    #/// Создать слой по значению параметра сигмоидальной функции
    #/// <param name="count">количество нейронов</param>
    #/// <param name="inputs_count">Количество входов</param>
    #/// <param name="param">Значение параметра сигмоидальной функции</param>
    def create_layer(self,count, inputs_count, param):
        sg=SignFunc()
        sg.par=param
        layer = Layer(count, inputs_count, sg)
        self.layers.append(layer)

    #Печатать нейрость
    def print(self):
        print("------")
        count=len(self.layers)
        i=0
        while i<count:
            ncount=len(self.layers[i].neurons)
            print("Слой ",i," Нейронов: ",)
            j=0
            while j<ncount:
                neuron=self.layers[i].neurons[j]
                print("  Нейрон ",j," входов ",len(neuron.inputs)," весов ",len(neuron.weights)," trans ", neuron.trans.get_id())
                icount=len(neuron.inputs)
                k=0
                while k<icount:
                    print("    вход ",k,"=",neuron.inputs[k])
                    k=k+1
                j=j+1
            i=i+1    
        print("------")

#Класс обучающей выборки
class StudyMatrixItem:

    #Конструктор
    #a_incomes - входы
    #a_outcomes - выходы
    def __init__(self,a_incomes, a_outcomes):
        self.incomes=a_incomes #входы
        self.outcomes=a_outcomes #выходы
            
#Класс генетического алгоритма
class GeneticAlgorith:

    #Конструктор
    def __init__(self):
    
        #Минимальное количество особей в популяции (если окажется меньше этого числа то селекцию не делаем)
        self.min_count = 5

        #Максимальное количество особей в популяции (если окажется больше этого числа то обрезаем до этого числа)
        self.max_count = 30

        #Начальное количество мест в популяции
        self.count=10

        #Разрешить мутацию замены передаточной функции
        self.allow_change_trans_function=True

        self.p_mutation=0.1 #Вероятность мутации
        self.population=[] #популяция
        self.selection=[] #Обучающая выборка
        self.testing=[] #Тестировочная выборка

        self.level=0.5 #Уровень мутации

    #Инициализировать из нейросети
    #net - нейросеть
    #count_source_mutation - Количество мутаций исходной нейросети в начальной выборкe
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


    #/// Размножение
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

    #/// Вычислить вероятность скрещивания
    #/// <param name="net1">Первая нейросеть (номер)</param>
    #/// <param name="net2">Вторая нейросеть (номер)</param>
    #/// <param name="count">Количество "особей" в популяции</param>
    #/// <returns>Вероятность скрещивания</returns>
    def cross_probability(self,net1, net2, count):
        return (2.0*float(count)-float(net1)-float(net2)) / (2.0*float(count)-1.0)
            
    #/// Скрещивание нейронных сетей
    #/// <param name="net1">Первая нейросеть</param>
    #/// <param name="net2">Вторая нейросеть</param>
    #/// <returns>Результат скрещивания</returns>
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
      

    #Осуществить селекцию
    def selecticting(self):
        if len(self.population)<self.min_count:
            return
        items_for_removed=[]

        #Начинаем со второго элемента (индекс 1), так как первый (индекс 0) должен выжить в любом случае)
        while len(self.population)>self.max_count:
            i = 1
            _count=float(len(self.population))
            while i < len(self.population):
                if rnd.random()<float(i)/_count:
                    items_for_removed.append(self.population[i])
                i=i+1

            #удаляем выбранные для удаления
            for item in items_for_removed:
                self.population.remove(item)

            #очистим список, чтобы не удалять второй раз
            items_for_removed.clear()

    #Сортировка по передаточной функции
    @staticmethod
    def sort_by_target_function(net):
        if net.target_function==False:
            return 99999999999999
        else:
            return net.target_function

    #Установить входные данные
    #<param name="net">Нейросеть (GANeuralNet)</param>
    #<param name="item">Входные данные (StudyMatrixItem)</param>
    def set_incomes(self, net, item):
        net.set_incomes(item.incomes)

    #Вычислить целевую функцию   
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
              
    #Сортировка популяции
    def sorting(self):
                    
        #сначала надо вычислить целевую функцию
        i = 1
        while i < len(self.population):
            self.calk_target_function(self.population[i])
            i=i+1

        #сортируем
        self.population.sort(key=self.sort_by_target_function)

        self.best_net = self.population[0]                  

    #Следующая эпоха
    def next_age(self):
        self.prev_spec = self.population[0]
        self.reproduction()
        self.sorting()
        self.selecticting()