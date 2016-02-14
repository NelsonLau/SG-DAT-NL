t1 = (3,4,5)
t2 = (1,1,1)
x = [t1,t2]
solutions = [tri for tri in x if tri[0]**2 + tri[1]**2 == tri[2]**2]
print solutions