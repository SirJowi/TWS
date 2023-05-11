""" Programm zur Ermittlung der Versagenswahrscheinlichkeit eines statischen Systems im Zuge einer Übungsaufgabe
    03/05/23 """

import numpy as np
import math
from scipy.stats import norm
import scipy.integrate as integrate

# Last G (Normalverteilung) --------------------------------------------------------------------------------------------
my_x1 = 230.0     # Erwartungswert der Last G in [kN]
sigma_x1 = 50.0   # Standardabweichung der Last G in [kN]


def f1(x1):      # Wahrscheinlichkeitsdichtefunktion
    return 1 / np.sqrt(2 * math.pi * sigma_x1 ** 2) * np.exp(-1/2 * ((x1 - my_x1)/sigma_x1)**2)


# Festigkeit f_y (Logarithmisch Normalverteilt) ------------------------------------------------------------------------
my_x2 = 25.7 * 10**4    # Erwartungswert der Festigkeit in [kN/m^2]
sigma_x2 = 2.01 * 10**4     # Standardabweichung der Festigkeit [kN/m^2]
x_02 = 19.95 * 10**4    # [kN/m^2]
# Parameter der Wahrscheinlichkeitsverteilung
sigma_u = np.sqrt(np.log(1+(sigma_x2/(my_x2-x_02))**2))
my_u = np.log(my_x2-x_02)-(sigma_u**2)/2


# Transformation LNV -> SNV
def SNV(x2):
    if x2 > x_02:
        y = (np.log(x2 - x_02) - my_u) / sigma_u
        val = norm.cdf(y)
    else:
        val = 0
    return val


# Grenzzustandsfunktion ------------------------------------------------------------------------------------------------
A = 0.014   # [m^2] Querschnittsfläche
W_el = 1124 * 10**-6    # [m^3]
W_pl = 1350 * 10**-6    # [m^3]
C1_el = -1 * ((29/97 * np.sin(np.arctan(5.2/3.9))) / A + (29/97 * 3.9) / W_el)     # Grenzzustand elastisch
C1_pl = -1 * ((29/97 * 3.9) / W_pl)    # Grenzzustand plastisch


# Versagenswahrscheinlichkeit ------------------------------------------------------------------------------------------
def P_f(x_1):
    x_2 = - C1_el * x_1
    return f1(x_1) * SNV(x_2)


P_f_int = integrate.quad(P_f, -x_02/C1_el, 10**5)

# Ausgabe --------------------------------------------------------------------------------------------------------------
print("sigma:", sigma_u)
print("my:", my_u)
print("C1_el:", C1_el)
print("C1_pl:", C1_pl)
print("P_f:", P_f_int)
