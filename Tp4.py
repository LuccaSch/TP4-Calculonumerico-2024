import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

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

plt.plot(x_new, y_newton, 'y--', label='Trayectoria interpolada (Newton)', linewidth=2)
plt.plot(distanciaX, alturaY, 'ro', label='Puntos críticos', markersize=8)
# Anotaciones en los puntos críticos
for i in range(len(distanciaX)):
    plt.text(distanciaX[i], alturaY[i] + 1, f"({distanciaX[i]}, {alturaY[i]})", fontsize=10, ha='center')

# Configuraciones de la gráfica
plt.title('Diseño de la montaña rusa con Polinomio de Interpolación de Newton')
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.legend()
plt.grid(True)
plt.show()

#Opcion 2 Splaine

# Crear el spline cúbico con el typo natural que define las derivadas en los extremos como 0
cs = CubicSpline(distanciaX, alturaY, bc_type='natural')

y_spl = cs(x_new)

# Graficar los puntos críticos y la curva interpolada
plt.plot(x_new, y_spl, 'b--', label='Trayectoria suavizada (Spline cúbico)', linewidth=2)
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

#---------------------------------------------1)D---------------------------------------------
#Comparación de Suavidad: Método de Energía
# Cálculo de la segunda derivada para ambas curvas
# Derivada de la curva de Newton (segunda derivada numérica)
dydx_newton = np.gradient(y_newton, x_new)
d2ydx2_newton = np.gradient(dydx_newton, x_new)

# Derivada de la curva Spline (usamos la función spline para derivar)
d2ydx2_spline = cs(x_new, 2)

# Cálculo de la energía para ambas curvas
# Usamos el espaciado entre puntos (dx) en lugar de pasar x_new directamente
dx = x_new[1] - x_new[0]  

energy_newton = simpson(d2ydx2_newton**2, dx=dx)
energy_spline = simpson(d2ydx2_spline**2, dx=dx)

print(f"Energía de la curva de Newton: {energy_newton}")
print(f"Energía de la curva Spline: {energy_spline}")

# Comparación final
if energy_newton > energy_spline:
    print("La curva Spline es más suave que la de Newton.")
else:
    print("La curva de Newton es más suave que la Spline.")
plt.plot(x_new, y_newton, 'y--', label='Trayectoria interpolada (Newton)', linewidth=2)
plt.plot(x_new, y_spl, 'b--', label='Trayectoria suavizada (Spline cúbico)', linewidth=2)
plt.plot(distanciaX, alturaY, 'ro', label='Puntos críticos', markersize=8)
# Anotaciones en los puntos críticos
for i in range(len(distanciaX)):
    plt.text(distanciaX[i], alturaY[i] + 1, f"({distanciaX[i]}, {alturaY[i]})", fontsize=10, ha='center')

# Configuraciones de la gráfica
plt.title('Comparacion de los diseños de las montañas rusas con Polinomio de Interpolación de Newton y Spline Cúbico')
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.legend()
plt.grid(True)


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


#-------------------------------------------- 2)A ------------------------------------------------
# Parámetros del problema
n_supports = 20  # Número de soportes
min_distance = 10  # Distancia mínima entre soportes
max_distance = 20  # Distancia máxima entre soportes

# Crear la clase de fitness y la clase individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Definir el spline cúbico
distanciaX = np.array([0, 50, 100, 150, 200, 250, 300])
alturaY = np.array([10, 60, 55, 70, 40, 50, 30])
cs = CubicSpline(distanciaX, alturaY, bc_type='natural')

# Inicializar el toolbox
toolbox = base.Toolbox()

# Función para crear un individuo válido
def create_valid_individual():
    supports = [np.random.uniform(0, 20)]  # Primer soporte entre 0 y 20
    for _ in range(1, n_supports):
        next_support = supports[-1] + np.random.uniform(10, 20)  # Sumar entre 10 y 20 al anterior
        supports.append(next_support)
    return creator.Individual(np.sort(supports))  # Asegurarse de que los soportes estén ordenados

# Registrar la nueva función en el toolbox
toolbox.register("individual", create_valid_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

# Función de evaluación
def eval_supports(individual):
    heights = cs(individual)  # Alturas en los puntos de soporte
    return (np.sum(heights),)  # Devolver una tupla (DEAP requiere)

toolbox.register("evaluate", eval_supports)

# Función para validar que un individuo cumpla las restricciones
def valid_individual(individual):
    distances = np.diff(np.sort(individual))
    return np.all(distances >= min_distance) and np.all(distances <= max_distance)

# Función para corregir individuos inválidos
def correct_invalid_individual(individual):
    valid = False
    while not valid:
        # Crear un nuevo individuo válido
        individual = create_valid_individual()
        valid = valid_individual(individual)
    return individual

# Función de mutación
def mutate(individual):
    index = np.random.randint(0, n_supports)  # Elegir un índice aleatorio
    change = np.random.uniform(-5, 5)  # Cambio aleatorio en el soporte
    new_value = individual[index] + change

    # Asegurarse de que el nuevo valor está dentro de los límites
    if new_value < 0:
        new_value = 0
    individual[index] = new_value
    # Validar y corregir si el individuo no es válido
    if not valid_individual(individual):
        individual[:] = correct_invalid_individual(individual)

# Registrar la función de mutación en el toolbox
toolbox.register("mutate", mutate)

# Algoritmo evolutivo
def genetic_optimization():
    population = toolbox.population(n=300)  # Tamaño de la población
    ngen = 50  # Número de generaciones
    cxpb, mutpb = 0.5, 0.2  # Probabilidades de cruce y mutación

    for gen in range(ngen):

        # Evaluar toda la población
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Selección de los mejores individuos basados en la sumatoria de alturas
        selected = tools.selBest(population, k=len(population)//2)  # Seleccionar la mitad de los mejores

        # Crear descendientes a partir de los mejores individuos
        offspring = list(map(toolbox.clone, selected))

        # Crossover: Hacer el cruce solo si hay un número par de descendientes
        for i in range(1, len(offspring), 2):
            if np.random.rand() < cxpb:
                toolbox.mate(offspring[i-1], offspring[i])
                # Validar y corregir si el individuo no es válido
                if not valid_individual(offspring[i-1]):
                    offspring[i-1] = correct_invalid_individual(offspring[i-1])
                if not valid_individual(offspring[i]):
                    offspring[i] = correct_invalid_individual(offspring[i])

        # Mutación
        for mutant in offspring:
            if np.random.rand() < mutpb:
                toolbox.mutate(mutant)

        # Evaluación de nuevos individuos
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Reemplazar la población por los descendientes
        # Mantener los mejores y reemplazar el resto
        population[:] = selected + offspring

    # Seleccionar el mejor individuo al final
    best_individual = tools.selBest(population, 1)[0]
    return best_individual

# Optimización
best_supports = genetic_optimization()



#----------Ayuda a las Conclusiones----------

# Calcular la suma de las alturas y el promedio del mejor individuo
heights = cs(best_supports)
sum_heights = np.sum(heights)
print(f"Cantidad de material total para el mejor individuo: {sum_heights}")

prom_heights = np.mean(heights)
print(f"Sumatoria promedio de las alturas del mejor individuo: {prom_heights}")

# calcular el promedio de las alturas del spline total
x_new = np.linspace(0, 346, 1000)  
y_spl = cs(x_new)  

# Calcular el promedio de las alturas
average_height = np.mean(y_spl)

print(f"El promedio de las alturas del spline es: {average_height:.2f} metros")

#----------GRAFICA----------

# Generar la gráfica de la trayectoria suavizada
x_new = np.linspace(0, 346, 500)
y_spl = cs(x_new)

# Graficar la trayectoria del spline cúbico
plt.plot(x_new, y_spl, 'b-', label='Trayectoria suavizada (Spline cúbico)', linewidth=2)

# Graficar los soportes como cuadrados grises
plt.plot(best_supports, cs(best_supports), 's', color='gray', label='Soportes optimizados', markersize=8)

# Añadir líneas punteadas desde y=10 hasta cada soporte
for soporte in best_supports:
    plt.plot([soporte, soporte], [10, cs(soporte)], 'k--')  # Línea punteada entre y=10 y la altura del soporte

# Configuraciones de la gráfica
plt.title('Optimización de Soportes de la Montaña Rusa')
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()
#-------------------------------------------- 2)B ------------------------------------------------

# Cálculo de la curvatura del spline cúbico
curvatura_spline = cs.derivative(nu=2)

# Función para calcular el promedio ponderado de las alturas usando la curvatura como peso
def weighted_average_heights(supports, spline, curvatura):
    alturas = spline(supports)
    curvaturas = np.abs(curvatura(supports))  # Obtener las curvaturas absolutas en los puntos de soporte
    weighted_sum = np.sum(alturas * curvaturas)  # Sumar las alturas ponderadas por las curvaturas
    total_weight = np.sum(curvaturas)  # Sumar todas las curvaturas (los pesos)
    
    return weighted_sum / total_weight  # Calcular el promedio ponderado

# Modificar la función de evaluación para que use el promedio ponderado de alturas
def eval_supports_weighted(individual):
    return (weighted_average_heights(individual, cs, curvatura_spline),)

# Registrar la nueva función de evaluación en el toolbox
toolbox.register("evaluate", eval_supports_weighted)

# Ejecutar la optimización genética con la nueva función de evaluación
best_supports_weighted = genetic_optimization()

# Calcular la suma de las alturas ponderadas y el promedio del mejor individuo
heights_weighted = cs(best_supports_weighted)
curvature_weighted = np.abs(curvatura_spline(best_supports_weighted))

# Imprimir resultados finales
weighted_sum_heights = np.sum(heights_weighted * curvature_weighted)
print(f"Cantidad de material total ponderado para el mejor individuo: {weighted_sum_heights:.2f}")
cantidad_material=np.sum(best_supports_weighted)
print(f"Cantidad de material total para el mejor individuo: {cantidad_material:.2f}")

weighted_average_heights_result = np.mean(heights_weighted * curvature_weighted / np.sum(curvature_weighted))
print(f"Sumatoria promedio ponderada de las alturas del mejor individuo: {weighted_average_heights_result:.2f}")

#---------- GRAFICA DEL MEJOR RESULTADO PONDERADO ----------
# Generar la gráfica de la trayectoria suavizada
x_new = np.linspace(0, 346, 500)
y_spl = cs(x_new)

# Graficar la trayectoria del spline cúbico
plt.plot(x_new, y_spl, 'b-', label='Trayectoria suavizada (Spline cúbico)', linewidth=2)

# Graficar los soportes como cuadrados grises
plt.plot(best_supports_weighted, cs(best_supports_weighted), 's', color='gray', label='Soportes optimizados (Ponderado)', markersize=8)

# Añadir líneas punteadas desde y=10 hasta cada soporte
for soporte in best_supports_weighted:
    plt.plot([soporte, soporte], [10, cs(soporte)], 'k--')  # Línea punteada entre y=10 y la altura del soporte

# Configuraciones de la gráfica
plt.title('Optimización de Soportes Ponderada por Curvatura de la Montaña Rusa')
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()


# Función para calcular la segunda derivada del polinomio de Newton en los puntos de soporte
def segunda_derivada_newton(x, x_data, dd):
    n = len(x_data)
    resultado = np.zeros_like(x, dtype=float)
    
    # Calcular la segunda derivada (esencialmente buscando el término de orden 2 en adelante)
    for i in range(2, n):
        termino = dd[0, i]  # Término de la tabla de diferencias divididas
        for j in range(i):
            termino *= (x - x_data[j])  # Multiplicando por los factores (x - x_data[j])
        resultado += termino

    return resultado
    
# Crear tabla de diferencias divididas para el polinomio de Newton
tabla_divididas = diferencias_divididas(distanciaX, alturaY)

# Función para calcular el promedio ponderado de las alturas usando la curvatura como peso para Newton
def weighted_average_heights_newton(supports, x_data, dd):
    alturas = np.abs(polinomio_newton(supports, x_data, dd))  # Alturas del polinomio de Newton en los puntos de soporte
    curvaturas = np.abs(segunda_derivada_newton(supports, x_data, dd))  # Curvaturas absolutas (segunda derivada)
    weighted_sum = np.abs(np.sum(alturas * curvaturas))  # Sumar las alturas ponderadas por las curvaturas
    total_weight = np.sum(curvaturas)  # Sumar todas las curvaturas (los pesos)
    
    return weighted_sum / total_weight  # Calcular el promedio ponderado


# Modificar la función de evaluación para usar el promedio ponderado de alturas con el polinomio de Newton
def eval_supports_weighted_newton(individual):
    return (weighted_average_heights_newton(individual, distanciaX, tabla_divididas),)

# Registrar la nueva función de evaluación en el toolbox para Newton
toolbox.register("evaluate", eval_supports_weighted_newton)

# Ejecutar la optimización genética con la nueva función de evaluación basada en Newton
best_supports_weighted_newton = genetic_optimization()


best_supports_weighted_newton = np.clip(best_supports_weighted_newton, distanciaX[0], distanciaX[-1])

# Calcular la suma de las alturas ponderadas y el promedio del mejor individuo con Newton
heights_weighted_newton = np.abs(polinomio_newton(best_supports_weighted_newton, distanciaX, tabla_divididas))
curvature_weighted_newton = np.abs(segunda_derivada_newton(best_supports_weighted_newton, distanciaX, tabla_divididas))

# Imprimir resultados finales para Newton
weighted_sum_heights_newton = np.sum(heights_weighted_newton * curvature_weighted_newton)
print(f"Cantidad de material total ponderado para el mejor individuo (Newton): {weighted_sum_heights_newton:.2f}")
cantidad_material=np.sum(best_supports_weighted_newton)
print(f"Cantidad de material total para el mejor individuo (Newton): {cantidad_material:.2f}")

weighted_average_heights_result_newton = np.mean(heights_weighted_newton * curvature_weighted_newton / np.sum(curvature_weighted_newton))
print(f"Sumatoria promedio ponderada de las alturas del mejor individuo (Newton): {weighted_average_heights_result_newton:.2f}")

#---------- GRAFICA DEL MEJOR RESULTADO PONDERADO (NEWTON) ----------
# Generar la gráfica de la trayectoria del polinomio de Newton
x_new = np.linspace(0, 303.3, 500)
y_newton = polinomio_newton(x_new, distanciaX, tabla_divididas)

plt.plot(x_new, y_newton, 'g-', label='Trayectoria interpolada (Newton)', linewidth=2)
# Graficar los soportes como cuadrados grises
plt.plot(best_supports_weighted_newton, polinomio_newton(best_supports_weighted_newton, distanciaX, tabla_divididas), 's', color='gray', label='Soportes optimizados (Newton)', markersize=8)

# Añadir líneas punteadas desde y=10 hasta cada soporte
for soporte in best_supports_weighted_newton:
    plt.plot([soporte, soporte], [10, polinomio_newton(soporte, distanciaX, tabla_divididas)], 'k--')  # Línea punteada entre y=10 y la altura del soporte



# Configuraciones de la gráfica
plt.title('Optimización de Soportes Ponderada por Curvatura (Newton)')
plt.xlabel('Distancia (m)')
plt.ylabel('Altura (m)')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()