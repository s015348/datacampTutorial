# From http://quant.jd.com/research/
# Import libraries
import numpy as np
import statsmodels.tsa.stattools as sts
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# Generating X, Y
X = np.random.randn(1500)
Y = np.random.randn(1500)

# Plot X,Y
plt.scatter(X,Y)
plt.show()
print("correlation of X and Y is ")
print(np.corrcoef(X,Y)[0,1])

# Plot X,Y with normal noises
X = np.random.randn(1500)
Y = X + np.random.normal(0,0.1,1500)

plt.scatter(X,Y)
plt.show()
print("correlation of X and Y is ")
np.corrcoef(X,Y)[0,1]

# Plot X,Y with poisson noises
X = np.random.randn(1500)
Y = X + np.random.poisson(size=1500)

plt.scatter(X,Y)
plt.show()
print("correlation of X and Y is ")
np.corrcoef(X,Y)[0,1]