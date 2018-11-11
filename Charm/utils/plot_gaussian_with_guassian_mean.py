import sys

import matplotlib.pyplot as plt
import numpy as np
from mcerp import *

u_f_0 = 0
s_f_0 = sys.argv[1]
delta_f = sys.argv[2]

print(u_f_0, s_f_0, delta_f)

n = 500 # sample size for meta distribution
m = 200 # sample size for generated distribution
fs = np.random.normal(u_f_0, s_f_0, n*m)

generated_us_f = np.random.normal(u_f_0, delta_f, n)
gen_fs = []
for u_f in generated_us_f:
    gen_fs.extend(np.random.normal(u_f, s_f_0, m))

print(norm.fit(gen_fs))

B = 500 
p, e = np.histogram(gen_fs, bins=B)
xs = e[:-1] + (e[1] - e[0])/2
center = (e[:-1] + e[1:])/2
width = 0.7 * (e[1] - e[0])
plt.figure()
plt.xlabel('f')
plt.ylabel('count')
plt.bar(center, p, align='center', width=width, color='blue', alpha=0.5)

B = 500 
p, e = np.histogram(fs, bins=B)
xs = e[:-1] + (e[1] - e[0])/2
center = (e[:-1] + e[1:])/2
width = 0.7 * (e[1] - e[0])
plt.bar(center, p, align='center', width=width, color='red', alpha=0.2)

plt.show()
