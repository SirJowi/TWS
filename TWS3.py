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
for i in range(len(A)):
    c_j.append(- abs(S[i] / ( A[i] * 10**-3 ) ))

# Versagenswahrscheinlichkeit P_f (keine Korrelation) ------------------------------------------------------------------
def Integrand(x1):
    faktor1 = 1.
    for i in range(len(c_j)):
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

#print("1-Integral über f_1max: ", 1-integrate.quad(exMax1,start,end)[0])
#print("LNV_cdf vom Startwert", LNV_cdf(start))
#print("Startwert des Integranten: ", Integrand(start))
#print("Endwert des Integranten: ", Integrand(end))
#print("Endwert des f_1max: ", exMax1(end))
#print("Endwert des F_2LNV: ", 1.-Integrand(end)/exMax1(end))
#print("Überlebenswahrscheinlichkeit: ", P_s)
print("------------------------------------------------------------")
print("AUFGABE A")
print("------------------------------------------------------------")
# Ergebnis der Versagenswahrscheinlichkeit mit Quad
print("Versagenswahrschenlichkeit\t:", P_f)

# Validierung des Integrationsergebnisses mit Simpson-Regel
print("Simpson, P_f\t\t\t\t:", 1 - P_s_simpson)

# Versagenswahrscheinlichkeit P_f (volle Korrelation) ------------------------------------------------------------------

# für jeden Stab einzeln die Versagenswahrscheinlichkeiten berechnen
print("")
print("------------------------------------------------------------")
print("AUFGABE B")
print("------------------------------------------------------------")
def Integrand1(x1):
    return LNV_cdf(-c_j[i] * x1) * exMax1(x1)
temp = 0
temp_i = 0
for i in range(10):
    if c_j[i] == 0.:
        continue
    else:
        P_f_j = integrate.quad(Integrand1, -x_02/c_j[i], 10000)[0]
        if P_f_j > temp:
            temp = P_f_j
            temp_i = i
        else:
            continue

print("Stab", temp_i+1,"maßgebend mit P_f\t:", temp)

print("")
print("------------------------------------------------------------")
print("AUFGABE C")
print("------------------------------------------------------------")
# Versagenswahrscheinlichkeit P_f (Knicken) ----------------------------------------------------------------------------

I = [1.46, 1.46, 1.46, 1.46, 1.46, 1.78, 1.46, 1.46, 1.46, 2.1]
l = [5.6, 4.2, 4.2, 7.0, 4.2, np.sqrt(35.28), 5.6, 4.2, 7.0, 4.2]
E = 2.10 * 10**8
k = l

for i in range(10):
    if S[i] < 0:    # nur für Druckstäbe sinnvoll
        k[i] = ( E * I[i] * 10**-5 * math.pi**2 ) / l[i]**2
        x1 = - k[i] / S[i]
        F_exMax1 = 1 - np.exp(-np.exp(-a * (x1 - b)))
        print("Stab:", i+1, "\tP_f_k:", F_exMax1)

