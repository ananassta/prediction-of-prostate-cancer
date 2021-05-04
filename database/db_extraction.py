import csv, pyodbc

# set up some constants
MDB = 'New База РПЖ.mdb'
DRV = '{Microsoft Access Driver (*.mdb)}'
PWD = 'pw'

# connect to db
con = pyodbc.connect('DRIVER={};DBQ={};PWD={}'.format(DRV,MDB,PWD))
cur = con.cursor()

# run a query and get the results
SQL = 'SELECT * FROM Final;' # your query goes here
rows = cur.execute(SQL).fetchall()
#print(rows)

# create some arrays for future db
id = []
age = []
data_of_birth = []
weight = []
height = []
bmi = []
prostate_volume = []
psa = []
density_psa = []
T = []
N = []
M = []
G = []

# set up some auxiliary variables
k = 0
sum_w = 0
sum_h = 0
sum_v = 0
sum_p = 0
id_index = True

# fill our arrays
for i in range(len(rows)):
    if id_index == True:
        id.append(rows[i][0])
    k = k+1
    sum_w = sum_w + rows[i][2]
    sum_h = sum_h + rows[i][3]
    sum_v = sum_v + rows[i][4]
    sum_p = sum_p + rows[i][5]
    if i != (len(rows)-1):
        if rows[i][0] == rows[i+1][0]:
            id_index = False
        if rows[i][0] != rows[i+1][0]:
            data_of_birth.append(rows[i][1])
            weight.append(sum_w/k)
            height.append(sum_h/k)
            prostate_volume.append(sum_v/k)
            psa.append(sum_p/k)
            T.append(rows[i][6])
            N.append(rows[i][7])
            M.append(rows[i][8])
            G.append(rows[i][9])
            id_index = True
            sum_w = 0
            sum_h = 0
            sum_v = 0
            sum_p = 0
            k = 0
    else:
        data_of_birth.append(rows[i][1])
        weight.append(sum_w / k)
        height.append(sum_h / k)
        prostate_volume.append(sum_v / k)
        psa.append(sum_p / k)
        T.append(rows[i][6])
        N.append(rows[i][7])
        M.append(rows[i][8])
        G.append(rows[i][9])
        id_index = True
        sum_w = 0
        sum_h = 0
        sum_v = 0
        sum_p = 0
        k = 0

# ghosting to a more "convenient" form and getting additional parameters
for i in range(len(N)):
    if N[i] == None:
        N[i] = '0'
    if N[i] == 'x':
        N[i] = '0'
    if M[i] == None:
        M[i] = '0'
    if id[i] == '51AF93A3-C251-43C0-A231-F8C52515124B':
        weight[i] = 71
        height[i] = 176
    if id[i] == 'CBEB6BB5-1ACD-4D38-ADE6-FE600D25BD4A':
        weight[i] = 64
        height[i] = 176
    if id[i] == 'F42CC368-32B9-4A95-AFF9-E22ADD767E1A':
        weight[i] = 97.2
        height[i] = 177
    density_psa.append(round(psa[i]/prostate_volume[i],5))
    bmi.append(round(weight[i]/((height[i]/100)*(height[i]/100)),5))
    psa[i]=round(psa[i],5)

# calculate age
SQL = 'SELECT [Главная].[N baza], [Главная].[дата рождения], [Главная].[дата уст д-за] FROM [Главная];' # your query goes here
rows = cur.execute(SQL).fetchall()

for i in range(len(id)):
    for j in range(len(rows)):
        if id[i] == rows[j][0]:
            birth = int(data_of_birth[i].year)
            diagnos = int(rows[j][2].year)
            age.append(diagnos-birth)

# checking our results
#print(id[0])

cur.close()
con.close()
