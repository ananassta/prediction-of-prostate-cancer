import genetic_algorithm as nl
import matplotlib.pyplot as plt
import db_execute as db
import numpy as np

# Normalize input data using minmax
# the gleason score will be normalized by multiplying by 0.1 later while filling in the training and test samples
min = db.bmi[0]
max = db.bmi[0]
min_psa = db.psa[0]
max_psa = db.psa[0]
min_pr = db.prostate_volume[0]
max_pr = db.prostate_volume[0]
min_age = db.age[0]
max_age = db.age[0]
for i in range(np.array(db.bmi).size):
  if db.bmi[i] > max:
    max = db.bmi[i]
  if db.bmi[i] < min:
    min = db.bmi[i]
  if db.age[i] > max_age:
    max_age = db.age[i]
  if db.age[i] < min_age:
    min_age = db.age[i]
  if db.psa[i] > max_psa:
    max_psa = db.psa[i]
  if db.psa[i] < min_psa:
    min_psa = db.psa[i]
  if db.prostate_volume[i] > max_pr:
    max_pr = db.prostate_volume[i]
  if db.prostate_volume[i] < min_pr:
    min_pr = db.prostate_volume[i]
d2 = 1
d1 = 0
bmi_norm = np.array(db.bmi)
age_norm = np.array(db.bmi)
psa_norm = np.array(db.bmi)
pr_v_norm = np.array(db.bmi)
for i in range(np.array(db.bmi).size):
  bmi_norm[i] = ((db.bmi[i] - min)*(d2-d1)/(max-min)) + d1
  age_norm[i] = (((db.age[i] - min_age)*(d2-d1))/(max_age-min_age)) + d1
  psa_norm[i] = ((db.psa[i] - min_psa)*(d2-d1)/(max_psa-min_psa)) + d1
  pr_v_norm[i] = ((db.prostate_volume[i] - min_pr)*(d2-d1)/(max_pr-min_pr)) + d1

# Сreate a genetic algorithm with a training sample
ga = nl.GeneticAlgorith()
for i in range(200):
    ga.selection.append(nl.StudyMatrixItem([age_norm[i], bmi_norm[i], pr_v_norm[i], psa_norm[i]], [db.G[i]*0.1]))

# Create a test sample
for i in range(200,243,1):
    ga.testing.append(nl.StudyMatrixItem([age_norm[i], bmi_norm[i], pr_v_norm[i], psa_norm[i]], [db.G[i]*0.1]))

# Create a neural network
net = nl.NeuralNet(4, 1)
#net.create_layer(5, 4, 100)
#net.create_layer(3, 4, 100)
net.create_layer(2, 4, 100)
net.create_layer(1, 2, 100)

c = 0

def magic(numList):
    s = ''.join(map(str, numList))
    return float(s)

# Displaying the test result
for item in ga.selection:
    net.set_incomes(item.incomes)
    net.compute()
    a = abs(magic(item.outcomes) - magic(net.outputs))
    b = (100 * a) / magic(item.outcomes)
    c = c + b
    print("Input: ", item.incomes, "; desired output: ", item.outcomes, " real output: ", net.outputs)
error_procent = c / 200
print("error: ", error_procent)

# Initializing genetic algorithm
ga.init_population_from_net(net, 3)

print("-------------")

# set initial arrays
x = []
y = []

# conduct training
i = 1
k = 1
x.append(0)
y.append(ga.population[0].target_function)
while i <= 500:
    ga.next_age()
    x.append(i)
    y.append(ga.population[0].target_function)
    i = i + 1
    k = k + 1

c = 0
# Displaying the test result
print("After optimization")
net = ga.population[0]
for item in ga.selection:
    net.set_incomes(item.incomes)
    net.compute()
    a = abs(magic(item.outcomes) - magic(net.outputs))
    b = (100 * a) / magic(item.outcomes)
    c = c + b
    print("Input: ", item.incomes, "; desired output: ", item.outcomes, " real output: ", net.outputs)
error_procent = c / 200
print("error: ", error_procent)

print("-------------")
print("Sample test results")
c = 0
for item in ga.testing:
    net.set_incomes(item.incomes)
    net.compute()
    a = abs(magic(item.outcomes) - magic(net.outputs))
    b = (100 * a) / magic(item.outcomes)
    c = c + b
    print("Input: ", item.incomes, "; desired output: ", item.outcomes, " real output: ", net.outputs)
error_procent = c / 43
print("error: ", error_procent)

# Building a plot
fig = plt.figure()
plt.plot(x, y)

# Displaying Axis Titles and Labels
plt.title('Function graph')
plt.ylabel('Axis Y')
plt.xlabel('Axis X')
plt.grid(True)
plt.show()
