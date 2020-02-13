import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

f = open("data_self_distillation.txt", "r")
f2 = open("data_no_distillation.txt", "r")
distill_acc_q = []
distill_acc_r = []
no_distill_acc_q = []
no_distill_acc_r = []

for x in f:
  x = x.split()
  if (x[1] == 'val') and (x[3] == '2') and (x[4] == '2'):
      distill_acc_q.append(1-float(x[-1]))
  elif (x[1] == 'val') and (x[3] == '32') and (x[4] == '32'):
      distill_acc_r.append(1-float(x[-1]))

for x in f2:
  x = x.split()
  if (x[1] == 'val') and (x[3] == '2') and (x[4] == '2'):
      no_distill_acc_q.append(1-float(x[-1]))
  elif (x[1] == 'val') and (x[3] == '32') and (x[4] == '32'):
      no_distill_acc_r.append(1-float(x[-1]))

fig = plt.figure()
ax = plt.axes()

# Moving average
for i in range(len(distill_acc_q)-2):
    distill_acc_q[i] = (distill_acc_q[i] + distill_acc_q[i + 1] + distill_acc_q[i + 2])/3
    distill_acc_r[i] = (distill_acc_r[i] + distill_acc_r[i + 1] + distill_acc_r[i + 2])/3

    no_distill_acc_q[i] = (no_distill_acc_q[i] + no_distill_acc_q[i + 1] + no_distill_acc_q[i + 2])/3
    no_distill_acc_r[i] = (no_distill_acc_r[i] + no_distill_acc_r[i + 1] + no_distill_acc_r[i + 2])/3

ax.plot(distill_acc_r)
ax.plot(distill_acc_q)
ax.plot(no_distill_acc_r, linestyle='dashed')
ax.plot(no_distill_acc_q, linestyle='dashed')

ax.set_xlabel('Epoch', fontsize=16)
ax.set_ylabel('Validation Accuracy (%)', fontsize=16)
ax.legend(['W:32 A:32 - With SD', 'W:2  A:2 - With SD',
           'W:32 A:32 - Without SD', 'W:2  A:2 - Without SD'], loc='lower right', fontsize=14)
plt.savefig('self_distillation.pdf')
plt.show()
