import numpy as np
minShootDist=5
maxShootDist=10
a=[]
for x in range(81):
    for y in range(160):
        if ((58 - x) ** 2 + (159 - y) ** 2 >= minShootDist ** 2 and (58 - x) ** 2 + (159 - y) ** 2 <= maxShootDist ** 2 and y <= x + 101 and y <= -x + 217):
            if x != 58:
                facing=np.arctan((58-x)/(158-y))+np.pi/2
            else:
                facing=np.pi/2
            if facing<0:
                facing+=np.pi*2
            facing = int(facing*12/np.pi)
            if facing > 2:
                a.append((x,y,facing))
print(a)
print()
print(a[0][0])#a[np.random.choice(len(a))])
