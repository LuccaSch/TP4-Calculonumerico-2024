import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline


#---------------------------------------------1)A---------------------------------------------
def vandermondeConversor(x):
    n = len(x)
    vandermonde_matrix = np.zeros((n, n))
    
    # Llenar la matriz de Vandermonde manualmente
    for i in range(n):
        for j in range(n):
            vandermonde_matrix[i, j] = x[i]**(n-j-1)
    
    return vandermonde_matrix

def polinomio(x, coef):
    return sum(coef[i] * (x ** (6 - i)) for i in range(len(coef)))


# Puntos críticos
distanciaX = np.array([0, 50, 100, 150, 200, 250, 300])
alturaY = np.array([10, 60, 55, 70, 40, 50, 30])

# Normalizar las distancias
distanciaX_normalizada = distanciaX / max(distanciaX)

# Generar la matriz de Vandermonde con los valores normalizados
vandermonde_matrix = vandermondeConversor(distanciaX_normalizada)

# Resolver el sistema lineal para encontrar los coeficientes
coeficientesV = np.linalg.solve(vandermonde_matrix, alturaY)

# Crear nuevos puntos para graficar la curva suavizada y normalizada
x_new_normalizado = np.linspace(0, 1, 500)

# Evaluar el polinomio en los nuevos puntos (normalizados)
y_new = polinomio(x_new_normalizado, coeficientesV)

# Graficar los puntos críticos y la curva interpolada
# Denormalizar x_new para graficar
x_new = x_new_normalizado * max(distanciaX)

plt.plot(x_new, y_new, 'g--', label='Trayectoria interpolada (Polinomio grado 6 vandermonde)', linewidth=2) 
plt.plot(distanciaX, alturaY, 'ro', label='Puntos críticos', markersize=8)

# Anotaciones en los puntos críticos
for i in range(len(distanciaX)):
    plt.text(distanciaX[i], alturaY[i] + 1, f"({distanciaX[i]}, {alturaY[i]})", fontsize=10, ha='center')

# Configuraciones de la gráfica
plt.title('Diseño de la montaña rusa con Polinomio de grado 6')
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()

#---------------------------------------------1)B y C---------------------------------------------

numero_condicion = np.linalg.cond(vandermonde_matrix)

if (numero_condicion>=10000):
    print(f"matriz está mal condicionada, Número de condición de la matriz de Vandermonde: {numero_condicion}")
else:
    print(f"número de condición de la matriz aceptable puede, Número de condición de la matriz de Vandermonde: {numero_condicion}")

#planteamos 2 alternarivas newton y Splines

#Opcion 1 newton

def diferencias_divididas(x, y):
    n = len(x)
    dd = np.zeros((n, n))
    dd[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (x[i + j] - x[i])
    
    return dd

def polinomio_newton(x, x_data, dd):
    n = len(x_data)
    resultado = np.zeros_like(x, dtype=float)
    
    for i in range(n):
        termino = dd[0, i]
        for j in range(i):
            termino *= (x - x_data[j])
        resultado += termino
    
    return resultado

tabla_divididas = diferencias_divididas(distanciaX, alturaY)

x_new = np.linspace(0, 300, 500)
y_newton = polinomio_newton(x_new, distanciaX, tabla_divididas) 

# Graficar los puntos críticos y la curva interpolada
import matplotlib.pyplot as plt

plt.plot(x_new, y_newton, 'b--', label='Trayectoria interpolada (Newton)', linewidth=2)

# Configuraciones de la gráfica
plt.title('Diseño de la montaña rusa con Polinomio de Interpolación de Newton')
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.legend()
plt.grid(True)


#Opcion 2 Splaine

# Crear el spline cúbico con el typo natural que define las derivadas en los extremos como 0
cs = CubicSpline(distanciaX, alturaY, bc_type='natural')

y_spl = cs(x_new)

# Graficar los puntos críticos y la curva interpolada
plt.plot(x_new, y_spl, 'y--', label='Trayectoria suavizada (Spline cúbico)', linewidth=2)
plt.plot(distanciaX, alturaY, 'ro', label='Puntos críticos', markersize=8)

# Anotaciones en los puntos críticos
for i in range(len(distanciaX)):
    plt.text(distanciaX[i], alturaY[i] + 1, f"({distanciaX[i]}, {alturaY[i]})", fontsize=10, ha='center')

# Configuraciones de la gráfica
plt.title('Diseño de la montaña rusa con Spline Cúbico')
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()

#---------------------------------------------1)E---------------------------------------------

altura_75_newton = polinomio_newton(75, distanciaX, tabla_divididas)
altura_225_newton = polinomio_newton(225, distanciaX, tabla_divididas)

altura_75_spline = cs(75)
altura_225_spline = cs(225)

print(f"Altura a 75 metros con el polinomio de Newton: {altura_75_newton}")
print(f"Altura a 225 metros con el polinomio de Newton: {altura_225_newton}")

print(f"Altura a 75 metros con el spline cúbico: {altura_75_spline}")
print(f"Altura a 225 metros con el spline cúbico: {altura_225_spline}")