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

plt.plot(x_new, y_new, 'g--', label='Trayectoria interpolada (Polinomio grado 6)', linewidth=2) 
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