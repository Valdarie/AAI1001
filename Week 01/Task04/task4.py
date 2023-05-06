import numpy as np
import matplotlib.pyplot as plt

a = 1   # Set mean of normal distribution
s = 1   # Set std dev for normal distribution

# Density of normal distribution
def normal_pdf(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x-mu)**2/(2*sigma**2))


# Density of each component of mixture model
def f1(x, a, s):
    return 0.5 * (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-(x + a)**2 / (2 * s**2))


def f2(x, a, s):
    return 0.5 * (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-(x - a)**2 / (2 * s**2))


# Density of mixture model
def mixture_pdf(x, a, s):
    return f1(x, a, s) + f2(x, a, s)


# Mean of mixture model
def mixture_mean():
    mean1 = -a * 0.5    # Mean of 1st component of mixture model
    mean2 = a * 0.5     # Mean of 2nd component of mixture model
    return mean1 + mean2

print("Mean Of Mixture Model:", mixture_mean())

# Variance of mixture model
def mixture_variance():
    var1 = ((-a - mixture_mean())**2 * f1(0, a, s)) + ((a - mixture_mean())**2 * f2(0, a, s)) # Variance of 1st and 2nd component of mixture model
    var2 = ((-a - mixture_mean())**2 * f2(0, a, s)) + ((a - mixture_mean())**2 * f1(0, a, s))
    return var1 + var2

print("Variance Of Mixture Model:", mixture_variance())

x = np.linspace(-5, 5, 1000)    # Generate 1000 evenly spaced values between -5 and 5 for x
y = mixture_pdf(x, a, s)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Mixture of Two Normal Distributions')
plt.show()
