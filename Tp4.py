import numpy as np
import matplotlib.pyplot as plt

def vandermondeConversor(x):
    n = len(x)
    vandermonde_matrix = np.zeros((n, n))
    
    # Llenar la matriz de Vandermonde manualmente
    
    for i in range(n):
        for j in range(n):
            vandermonde_matrix[i, j] = x[i]**(n-j-1)
    
    return vandermonde_matrix

def polinomio(x,coef):
    return coef[6]*(x**6)+coef[5]*(x**5)+coef[4]*(x**4)+coef[3]*(x**3)+coef[2]*(x**2)+coef[1]*(x**1)+coef[0]*(x**0)

# Puntos críticos
distanciaX = np.array([0, 50, 100, 150, 200, 250, 300])
alturaY = np.array([10, 60, 55, 70, 40, 50, 30])

# Generar la matriz de Vandermonde
vandermonde_matrix = vandermondeConversor(distanciaX)
print("Matriz de Vandermonde:")
print(vandermonde_matrix)

# Resolver el sistema lineal para encontrar los coeficientes
coeficientesV = np.linalg.solve(vandermonde_matrix, alturaY)
print("Coeficientes del polinomio:")
print(coeficientesV)

# Crear un polinomio a partir de los coeficientes
#polinomio = np.poly1d(coeficientesV)


# Crear nuevos puntos para graficar la curva suavizada
x_new = np.linspace(0, 300, 500)

y_new = polinomio(x_new,coeficientesV)

# Graficar los puntos críticos y la curva interpolada
plt.plot(x_new, y_new, label='Trayectoria interpolada (Polinomio grado 6)', color='b', linewidth=2)
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