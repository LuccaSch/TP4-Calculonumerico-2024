# TP4-Calculonumerico-2024

# TP4-Calculonumerico-2024

A- Como estamos trabajanda con polinomios de alto grado y números grandes (distanciaX max 300), el cálculo de potencias altas produce números inmensos. EJ 300^6, esto genera
pérdida de precisión: Las operaciones con números muy grandes  llevan a imprecisiones debido a las limitaciones de la representación numérica en los tipos int y hasta long.
la inestabilidad numérica probocaba un polinomio incorrecto asi que normalizamos los datos para solucionar este problema, haciendo esto logro escalalar los datos para que estén dentro de un rango más manejable (entre 0-1). La idea es reducir los valores de distanciaX antes de elevarlos a potencias altas, evitando así los grandes números que causan inestabilidad y luego desnormalizarlo generando asi una curva correcta usando vander con un polinomio de grado 6 que pase por todos los ptos.

B y C- Sabemos que el número de condición de una matriz mide cuánto puede cambiar la solución de un sistema de ecuaciones lineales en respuesta a cambios en los datos. Un número de condición alto indica que la matriz mal condicionada, lo que puede generar inestabilidad numérica.
el número de condición puede calcularse fácilmente usando np.linalg.cond. de numpy
arroja un valor de 36061.16 lo cual es muy alto y debemos proponer una alternativa.

Proponemos dos alternativas Newton y Splines


*alternativa 1 Utilizamos newton en vez de lagrange debido a:

-Fácil de actualizar: Si se añaden nuevos puntos de datos, no es necesario reconstruir todo el polinomio. Solo se añade un nuevo término al polinomio existente, esto es util si durante la contruccion de la montaña se desean añadir puntos extra

-Más estable que Lagrange en términos de oscilaciones: Al usar diferencias divididas en lugar de una fórmula directa basada en los puntos, puede ser más estable.

'para realizarlo manualmente creamos dos funciones diferencias_divididas y el polinomio_newton'



*alternativa 2 utilizamos Splines debido a:

-Suavidad en la Interpolación: Los splines garantizan que la función interpolada sea suave en los puntos de interpolación. 

A diferencia de los polinomios de alto grado, los splines, especialmente los cúbicos, tienden a evitar el fenómeno de Runge. Esto se debe a que el spline se ajusta en segmentos

-Flexibilidad y Precisión:
Los splines permiten una aproximación muy precisa a los datos sin necesidad de usar polinomios de alto grado. Se ajustan a los datos en tramos, proporcionando una aproximación precisa y flexible que puede manejar datos con irregularidades.

-Añadir nuevos puntos a un spline no requiere rehacer toda la interpolación, sino que sólo se deben ajustar los segmentos afectados.

Su unica desventaja es la construcción puede ser más compleja que la de un polinomio de Newton, para muchos casos prácticos, pero eso se compenza a posteriori gracias a que el costo computacional de evaluar y ajustar splines es menor, más eficiente debido a la estructura segmentada.

'Para crear los splines cubicos utilizaremos la libreria interpolate de scipy, mas especificamente CubicSpline'

D y E- Para comparar la suavidad de la curva obtenida con splines cúbicos con la del polinomio resultante de newton en el diseño de una montaña rusa, es útil considerar tanto la visualización gráfica como los aspectos técnicos
Los polinomios interpoladores de alto grado, como el nuestro, muestran oscilaciones significativas entre los puntos críticos. Estas oscilaciones resultan en una trayectoria menos suave.
Los splines cúbicos tienden a ser mucho más suaves porque ajustan una función polinómica de grado 3 en cada intervalo entre puntos críticos. Esto asegura que la curva sea continua en términos de la función, así como en la primera y segunda derivada.
En la gráfica, el spline cúbico muestra una transición mucho más suave entre los puntos críticos, sin las oscilaciones que se pueden observar con el polinomio de alto grado, la implementacion de cada opcion dependera de las condiciones que se requieran para la montaña tanto de adrenalina para los usuarios como de seguridad. Si se busca un recorrido mas "amigable" con el cliente se debera usar el modelado con splines, en cambio si se quiere subidas y bajadas mas bruscas para una montaña mas extrema se deberia usar el modelo por interpolacion de Newton.
