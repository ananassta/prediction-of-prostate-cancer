import neural_network_ga as nl 
import matplotlib.pyplot as plt

#create a genetic algorithm with a training sample
ga=nl.GeneticAlgorith()
ga.selection.append(nl.StudyMatrixItem([0.2491,  0.75,    0.131,   0.855],[0.4]))
ga.selection.append(nl.StudyMatrixItem([0.2551,  0.5,     0.0699,  0.3],[0.8]))
ga.selection.append(nl.StudyMatrixItem([0.4343,  0.61,    0.01689, 0.44],[0.6]))
ga.selection.append(nl.StudyMatrixItem([0.2484,  0.52,    0.1606,  0.6],[0.9]))
ga.selection.append(nl.StudyMatrixItem([0.3842,  0.64,    0.04644, 0.27],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.3395,  0.71,    0.0338,  0.57],[0.6]))
ga.selection.append(nl.StudyMatrixItem([0.418,   0.64,    0.0351,  0.58],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2623,  0.51,    0.00603, 0.1],[0.9]))
ga.selection.append(nl.StudyMatrixItem([0.3288,  0.74,    0.0052,  0.34],[0.5]))
ga.selection.append(nl.StudyMatrixItem([0.2315,  0.64,    0.05,    0.32],[0.8]))
ga.selection.append(nl.StudyMatrixItem([0.2409,  0.61,    0.01512, 0.19],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2809,  0.77,    0.00707, 0.27],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.3351,  0.52,    0.052,   0.43],[0.6]))
ga.selection.append(nl.StudyMatrixItem([0.3088,  0.64,    0.0067,  0.42],[0.6]))
ga.selection.append(nl.StudyMatrixItem([0.2708,  0.64,    0.026,   0.71],[0.8]))
ga.selection.append(nl.StudyMatrixItem([0.2595,  0.66,    0.0335,  0.49],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.3153,  0.65,    0.03642, 0.29],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2547,  0.64,    0.15,    0.71],[0.8]))
ga.selection.append(nl.StudyMatrixItem([0.2454,  0.67,    0.0294,  0.36],[0.9]))
ga.selection.append(nl.StudyMatrixItem([0.2747,  0.62,    0.056,   0.37],[0.9]))
ga.selection.append(nl.StudyMatrixItem([0.2642,  0.68,    0.046,   0.44],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.3657,  0.6,     0.00954, 0.48],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2768,  0.62,    0.0351,  0.55],[0.8]))
ga.selection.append(nl.StudyMatrixItem([0.3729,  0.64,    0.0079,  0.51],[0.9]))
ga.selection.append(nl.StudyMatrixItem([0.2603,  0.69,    0.0046,  0.2],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.297,   0.62,    0.0738,  0.29],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.3211,  0.73,    0.0162,  0.15],[0.6]))
ga.selection.append(nl.StudyMatrixItem([0.2967,  0.68,    1.322,   0.95],[0.8]))
ga.selection.append(nl.StudyMatrixItem([0.2768,  0.83,    0.07,    0.34],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2806,  0.7,     0.01427, 0.39],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2565,  0.71,    0.0588,  0.47],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.1983,  0.73,    0.07609, 0.56],[0.8]))
ga.selection.append(nl.StudyMatrixItem([0.1785,  0.64,    0.04073, 0.32],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.3086,  0.6,     0.011,   0.09],[0.9]))
ga.selection.append(nl.StudyMatrixItem([0.3241,  0.64,    0.0494,  0.46],[0.6]))
ga.selection.append(nl.StudyMatrixItem([0.2312,  0.63,    0.04425, 0.54],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2231,  0.67,    0.136,   0.62],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.3047,  0.5,     0.0699,  0.3],[0.8]))
ga.selection.append(nl.StudyMatrixItem([0.262,   0.64,    0.02049, 0.15],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2778,  0.68,    0.756,   0.35],[1.0]))
ga.selection.append(nl.StudyMatrixItem([0.2841,  0.55,    0.02102, 0.47],[0.5]))
ga.selection.append(nl.StudyMatrixItem([0.2311,  0.73,    0.00656, 0.32],[0.9]))
ga.selection.append(nl.StudyMatrixItem([0.3265,  0.59,    0.1649,  0.18],[0.6]))
ga.selection.append(nl.StudyMatrixItem([0.3737,  0.69,    0.16,    0.28],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2806,  0.7,     0.01427, 0.39 ],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2714,  0.6,     0.124,   0.71],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.2442,  0.64,    1.129,   0.53],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.3156,  0.63,    0.0391,  0.34],[0.7]))
ga.selection.append(nl.StudyMatrixItem([0.3228,  0.61,    0.0083,  0.14],[0.8]))
ga.selection.append(nl.StudyMatrixItem([0.262,   0.63,    0.0082,  0.13],[0.6]))


#Create a test sample
ga.testing.append(nl.StudyMatrixItem([0.2264, 0.72, 0.546,  0.27],[0.6]))
ga.testing.append(nl.StudyMatrixItem([0.2779, 0.54, 0.0075, 0.46],[0.7]))
ga.testing.append(nl.StudyMatrixItem([0.3356, 0.6,  0.159,  0.54],[0.6]))
ga.testing.append(nl.StudyMatrixItem([0.2837, 0.6,  0.3317, 0.65],[0.8]))
ga.testing.append(nl.StudyMatrixItem([0.245,  0.66, 0.1784, 0.35],[0.7]))


#Create a neural network
net=nl.NeuralNet(4,1)
net.create_layer(5,4,100)
net.create_layer(3,5,100)
net.create_layer(2,3,100)
net.create_layer(1,2,100)

c=0
def magic(numList):
    s = ''.join(map(str, numList))
    return float(s)

#Displaying the test result
for item in ga.selection:
    net.set_incomes(item.incomes)
    net.compute()
    a = abs(magic(item.outcomes) - magic(net.outputs))
    b = (100*a)/magic(item.outcomes)
    c=c+b
    print("Input: ",item.incomes,"; desired output: ",item.outcomes," real output: ", net.outputs)
error_procent = c/50
print ("error: ", error_procent)

#Initializing genetic algorithm
ga.init_population_from_net(net,3)

print("-------------")

#set initial arrays
x=[]
y=[]

#conduct training
i=1
k=1
x.append(0)
y.append(ga.population[0].target_function) 
while i<=500:
    ga.next_age()
    x.append(i)
    y.append(ga.population[0].target_function)
    i=i+1
    k=k+1

c=0
#Displaying the test result
print("After optimization")
net=ga.population[0]
for item in ga.selection:
    net.set_incomes(item.incomes)
    net.compute()
    a = abs(magic(item.outcomes) - magic(net.outputs))
    b = (100*a)/magic(item.outcomes)
    c=c+b
    print("Input: ",item.incomes,"; desired output: ",item.outcomes," real output: ", net.outputs)
error_procent = c/50
print ("error: ", error_procent)      

print("-------------")
print("Sample test results")
c=0
for item in ga.testing:
    net.set_incomes(item.incomes)
    net.compute()
    a = abs(magic(item.outcomes) - magic(net.outputs))
    b = (100*a)/magic(item.outcomes)
    c=c+b
    print("Input: ",item.incomes,"; desired output: ",item.outcomes," real output: ", net.outputs)
error_procent = c/5
print ("error: ", error_procent)   

#Building a plot
fig = plt.figure()
plt.plot(x, y)
     
#Displaying Axis Titles and Labels
plt.title('Function graph')
plt.ylabel('Axis Y')
plt.xlabel('Axis X')
plt.grid(True)
plt.show()