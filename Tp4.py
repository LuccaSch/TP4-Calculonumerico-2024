import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

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

