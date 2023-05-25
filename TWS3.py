""" Programm zur Ermittlung der Versagenswahrscheinlichkeit eines statischen Systems im Zuge einer Übungsaufgabe
    25/05/23 """

import numpy as np
import math
from scipy.stats import norm
import scipy.integrate as integrate


# Last G (Ex-max-Typ 1 [Gumbel]) ---------------------------------------------------------------------------------------
my_x1 = 410.0     # Erwartungswert der Last G in [kN]
sigma_x1 = 70.0   # Standardabweichung der Last G in [kN]

a = 1/sigma_x1 * math.pi / np.sqrt(6)

b = my_x1 - 0.5772 / a

def exMax1(x1):
    return a * np.exp(-a * (x1 - b) - np.exp(-a * (x1 - b)))


# Festigkeit f_y (Logarithmisch Normalverteilt) ------------------------------------------------------------------------
my_x2 = 30.2 * 10**4    # Erwartungswert der Festigkeit in [kN/m^2]
sigma_x2 = 2.44 * 10**4     # Standardabweichung der Festigkeit [kN/m^2]
x_02 = 19.9 * 10**4    # [kN/m^2]
# Parameter der Wahrscheinlichkeitsverteilung
sigma_u = np.sqrt(np.log(1+(sigma_x2/(my_x2-x_02))**2))
my_u = np.log(my_x2-x_02)-(sigma_u**2)/2


# Transformation LNV -> SNV
def LNV_cdf(x2):
    if x2 > x_02:
        y = (np.log(x2 - x_02) - my_u) / sigma_u
        val = norm.cdf(y)
    else:
        val = 0.
    return val

# Querschnittsparameter
A = [3.77, 3.77, 3.77, 3.77, 3.77, 4.7, 3.77, 3.77, 3.77, 5.74] # Querschnitt
S = [0, 1, 0, 1.25, -0.75, -np.sqrt(2), -1, 0.75, 0, -1.75] # Stabkräfte

c_j = []
for i in range(len(A) - 1):
    c_j.append(- abs(S[i] / ( A[i] * 10**-3 ) ))
print(c_j)
# Versagenswahrscheinlichkeit P_f
def Integrand(x1):
    faktor1 = 1.
    for i in range(len(c_j) - 1):
        faktor1 = faktor1 * (1 - LNV_cdf(-c_j[i] * x1))
    return faktor1 * exMax1(x1)

start = 0
end = 1000
P_s = integrate.quad(Integrand, start, end)
P_f = 1. - P_s[0]

x_list = np.linspace(0, 1000, 10000)
y_list = np.copy(x_list)

for i in range(len(x_list-1)):
    y_list[i] = Integrand(x_list[i])

P_s_simpson = integrate.simpson(y_list, x_list)

print("1-Integral über f_1max: ", 1-integrate.quad(exMax1,start,end)[0])
print("LNV_cdf vom Startwert", LNV_cdf(start))
print("Startwert des Integranten: ", Integrand(start))
print("Endwert des Integranten: ", Integrand(end))
print("Endwert des f_1max: ", exMax1(end))
print("Endwert des F_2LNV: ", 1.-Integrand(end)/exMax1(end))
print("Überlebenswahrscheinlichkeit: ", P_s)

# Ergebnis der Versagenswahrscheinlichkeit mit Quad
print("Versagenswahrschenlichkeit: ", P_f)

# Validierung des Integrationsergebnisses mit Simpson-Regel
print("Simpson, P_f:", 1 - P_s_simpson)