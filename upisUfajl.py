x1 = [0,2,3,4,5,5,6,6]
y1 = [0,2,3,4,544,5,6,6]

fp = open('podaci.txt', 'w')
fp.write('x1,y1\n')
for i in range(len(x1)):
    fp.write(f'{x1[i]},{y1[i]}\n')

