# From: http://matplotlib.org/users/pyplot_tutorial.html
import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)
plt.subplot(211)
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
#plt.show()

plt.figure(1)
plt.subplot(212)
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()



mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

# Working with text
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


# Controlling line properties
plt.plot(x, y, linewidth=2.0)
line, = plt.plot(x, y, '-')
line.set_antialiased(False) # turn off antialising

lines = plt.plot(x1, y1, x2, y2)
# use keyword args
plt.setp(lines, color='r', linewidth=2.0)
# or MATLAB style string value pairs
plt.setp(lines, 'color', 'r', 'linewidth', 2.0)