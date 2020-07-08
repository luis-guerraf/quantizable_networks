import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

f = open("data_regular.txt", "r")
f2 = open("data_no_switchable_batchnorm.txt", "r")
acc_1 = []
acc_2 = []
acc_3 = []
acc_4 = []
acc_1_non = []
acc_2_non = []
acc_3_non = []
acc_4_non = []

for x in f:
    x = x.split()
    if (x[1] == 'val') and (x[3] == '2') and (x[4] == '2'):
        acc_1.append(1-float(x[-1]))
    elif (x[1] == 'val') and (x[3] == '32') and (x[4] == '2'):
        acc_2.append(1-float(x[-1]))
    elif (x[1] == 'val') and (x[3] == '2') and (x[4] == '32'):
        acc_3.append(1-float(x[-1]))
    elif (x[1] == 'val') and (x[3] == '32') and (x[4] == '32'):
        acc_4.append(1-float(x[-1]))

for x in f2:
    x = x.split()
    if (x[1] == 'val') and (x[3] == '2') and (x[4] == '2'):
        acc_1_non.append(1-float(x[-1]))
    elif (x[1] == 'val') and (x[3] == '32') and (x[4] == '2'):
        acc_2_non.append(1-float(x[-1]))
    elif (x[1] == 'val') and (x[3] == '2') and (x[4] == '32'):
        acc_3_non.append(1-float(x[-1]))
    elif (x[1] == 'val') and (x[3] == '32') and (x[4] == '32'):
        acc_4_non.append(1-float(x[-1]))

fig = plt.figure()
ax = plt.axes()

# Moving average
for i in range(len(acc_1)-2):
    acc_1[i] = (acc_1[i] + acc_1[i + 1] + acc_1[i + 2])/3
    acc_2[i] = (acc_2[i] + acc_2[i + 1] + acc_2[i + 2])/3
    acc_3[i] = (acc_3[i] + acc_3[i + 1] + acc_3[i + 2])/3
    acc_4[i] = (acc_4[i] + acc_4[i + 1] + acc_4[i + 2])/3

for i in range(len(acc_1_non) - 2):
    acc_1_non[i] = (acc_1_non[i] + acc_1_non[i + 1] + acc_1_non[i + 2])/3
    acc_2_non[i] = (acc_2_non[i] + acc_2_non[i + 1] + acc_2_non[i + 2])/3
    acc_3_non[i] = (acc_3_non[i] + acc_3_non[i + 1] + acc_3_non[i + 2])/3
    acc_4_non[i] = (acc_4_non[i] + acc_4_non[i + 1] + acc_4_non[i + 2])/3

ax.plot(acc_4)
ax.plot(acc_3)
ax.plot(acc_2)
ax.plot(acc_1)
ax.plot(acc_4_non[0:37], linestyle='dashed')
ax.plot(acc_3_non[0:37], linestyle='dashed')
ax.plot(acc_2_non[0:37], linestyle='dashed')
ax.plot(acc_1_non[0:37], linestyle='dashed')

ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('Validation Accuracy (%)', fontsize=16)
ax.legend(['W:32 A:32 - With S-BN', 'W:2 A:32 - With S-BN',
           'W:32 A:2 - With S-BN', 'W:2 A:2 - With S-BN',
           'W:32 A:32 - Without S-BN', 'W:2 A:32 - Without S-BN',
           'W:32 A:2 - Without S-BN', 'W:2 A:2 - Without S-BN'],
          loc='lower right', fontsize=14, frameon=True)
plt.savefig('non_switchable_batchnorm.pdf')
plt.show()
