# ensamble.jl
Automatic Test Assembly (ATA)  in julia and a MIP solver

## Presentación {#presentación}

En la tradición psicométrica de los exámenes estandarizados, una versión de prueba es suficiente. Esta es "la" prueba, resultado de meses o años de estudios, de intentos y de errores. Cuando "la" prueba finalmentes está lista, se aplica a una población de referencia y se establece un estándar, la "norma", que servirá para clasificar a quienes la usan con respecto a esa población original. Sin embargo, hay muchas razones que inhiben esta práctica cuando se trabaja en un contexto educativo. Imaginemos lo que pasaría si usáramos la misma versión de prueba siempre; que con ella evaluáramos a los estudiantes que egresan cada año o a los que quieren ingresar en cada generación.

Ensamblar dos o más versiones de una prueba es exponencialmente más complicado que hacer una sola. No solamente hay que completar correctamente la tabla de contenidos conforme al diseño, lo cual en sí mismo controvertido, sino que también hay que procurar que las dificultades de sus reactivos sean comparables entre sí. De otra manera, tendremos una prueba más fácil que otra, con consecuencias que pueden llegar a ser muy graves para los sustentantes. Sería muy poco equitativo que la calificación de cada uno dependiera de la versión que le tocó, pues es sencillamente injusto que un sustentante no logre entrar a la universidad porque respondió la versión difícil.

La Teoría de Respuesta al Ítem o al reactivo (IRT, por sus siglas en inglés o TRI en español) es una solución común para resolver esto, pues una vez que se conocen las características estadísticas de los reactivos a partir de ensayos anteriores, se puede anticipar cómo funcionarán en el futuro. Con esto se puede ensamblar una versión con una idea más o menos clara de cómo funcionará. La Teoría Clásica de las pruebas puede servir, pero es más difícil contar con estimaciones de dificultad en la misma escala, por lo que el resultado puede diferir de lo esperado. Una vez calibrado el banco de reactivos, en un proceso de prueba y error, se van combinando hasta llegar al objetivo de tener versiones iguales dentro de cierto margen de tolerancia.

Para lograr versiones similares, el procedimiento tradicional es iterativo, combinando los reactivos disponibles en el banco, como piezas de rompecabezas, de acuerdo con sus contenidos y dificultades previamente estimadas, hasta lograr el objetivo.

Aquí también las computadoras han traído ventajas. El proceso manual es tedioso; puede tardar varias horas y en algunos casos hasta días, especialmente cuando se intenta ensamblar varios cuadernillos con decenas de reactivos cada uno, poniendo y quitando hasta lograr lo que se busca. Para facilitar el trabajo, y de paso minimizar los errores que resultan de hacerlo a mano, de un tiempo para acá se ha aprovechado el  poder de los métodos de "optimización" que hacen posible hacer el ensamble de las versiones de una manera relativamente automática[^fn:1].

La idea es que si el elaborador establece de manera detallada las características que se desean en la prueba, incluyendo los temas y contenidos necesarios, se puede dejar que la máquina elija entre las mejores combinaciones.


## Ventajas del ensamble automático {#ventajas-del-ensamble-automático}

-   El ensamble en sí mismo se convierte en un procedimiento mucho más rápido y menos tedioso.

-   Obliga a contar con especificaciones correctamente definidas y detalladas. Esto hace el proceso más objetivo y reproducible.

-   Permite ensamblar varias versiones simultáneamente, en vez de tener que comparar una a una. Esto también promueve un mejor aprovechamiento de los reactivos.

-   El proceso es lo bastante poderoso como para cumplir _objetivamente_ con _todas las restricciones_, incluso con tablas de especificaciones complejas.

-   Es posible ensamblar versiones específicas según las necesidades de cada aplicación. En algunos casos, incluso es posible crear y modificar la versión "al vuelo" para cada sustentante, es decir, adaptando el contenido de la prueba conforme se responde.

-   Hace posible contar también con reportes de ensamble detallados y rápidos de un modo también automático.


## Desventajas {#desventajas}

-   Aunque la tabla de especificaciones y las reglas de ensamble deben existir siempre, a veces no se hace así o se deja, indebidamente, un margen de maniobra para el elaborador de la prueba. Aunque esto no es "objetivo", muchas personas encargadas de ensamblar lo aprecian y consideran que es un arte. En el caso del ensamble automático no hay este margen y debe hacerse un esfuerzo significativo para contar con todas las reglas, bien detalladas y codificadas, para que la máquina las procese y la técnica funcione. Incluso las tolerancias deben ser explícitas.

-   Siempre es posible que el programa no encuentre una solución satisfactoria, sobre todo cuando las condiciones de ensamble son muy complejas o las reglas resultan contradictorias, como sucede muchas veces. En estos casos, hace falta de un esfuerzo adicional para identificar y corregir los problemas después de cada intento fallido.

-   Al igual que el proceso manual, las versiones finales no siempre se comportan exactamente como se espera en la aplicación operativa. El ensamble, automático o manual, no es una solución mágica e infalible, y en todos los casos hay que llevar a cabo una comprobación después de la aplicación, así como los procedimientos de escalamiento o equiparación de rutina.


## El software {#el-software}

Aquí se presenta un ejemplo de cómo puede hacerse esto con software gratuito. Por ahora no es más que un ejemplo y no se recomienda usar.


### Software {#software}


#### Para calibrar el banco {#para-calibrar-el-banco}

Esto no es estrictamente parte del proceso de ensamblado, pero es un requisito básico; no se puede seguir adelante sin un banco de reactivos bien calibrado con muestras suficientes. Para esto se puede utilizar cualquiera de los varios paquetes especializados para la calibración de reactivos. Algunos cuentan con una larga historia en el mercado, como Winsteps, Facets o ConQuest para la familia de modelos de Rasch. En el ámbito de la Teoría de Respuesta al Item están, entre otros, Bilog, Parscale, Multilog o FlexMIRT.  Incluso hay un número creciente de paquetes para sustituirlos y explorar nuevos modelos, principalmente los elaborados para el lenguaje estadístico R, como TAM, SIRT, MIRT, ErM y una larga lista siempre en aumento que puede encontrarse en [CRAN](https://cran.r-project.org/web/views/Psychometrics.html).


#### Para la optimización {#para-la-optimización}

Existe software especializado en la optimización que puede adaptarse con cierta facilidad a nuestro caso. La técnica más común se basa en la "programación mixta de enteros" (MIP en inglés), que es la que se utiliza aquí. Este software tiende a lograr las mejores combinaciones, aunque en casos complicados puede tardar más tiempo o no llegar a una solución. Es por eso que en algunos casos también se utilizan otras metodologías, como el "templado simulado" (_simulated annealing_), algoritmos genéticos, programación con restricciones (constraint programming), algoritmos para resolver flujos de redes (_network-flow problems_) e incluso otros que usan cadenas de Markov. Los programas de optimización lineal tienden a ser más precisos, sobre todo cuando se utilizan otras _restricciones_ (como incluir diversos temas, tipos de reactivos o cuando hay que considerar la existencia de reactivos _amigos_ y _enemigos_).

En este ejemplo se utiliza [Cbc](https://www.coin-or.org) (Coin-or branch and cut), un software gratuito, de código abierto y que ofrece una licencia muy permisiva. Algunos optimizadores comerciales suelen ser relativamente más poderosos, aunque tienden a ser costosos.


#### Julia y JuMP {#julia-y-jump}

Julia es un lenguage de alto nivel, que se compila a código muy eficiente y puede utilizarse interactivamente, lo que facilita el trabajo. Por su parte, JuMP es un paquete basado en Julia que ofrece herramientas para formular  modelos de optimización para usar con una larga lista de optimizadores. En este caso, los modelos se probaron con resultados casi indistintos con:

-   IBM CPLEX
-   Cbc (coin-or)
-   SICP
-   GLPK
-   HiGHS

También hay otras formas de hacer esto. En el lenguaje _R_, por ejemplo, el paquete TestDesign parece ser una buena solución que ahorra varios pasos y no requiere demasiada programación. Otros ejemplos de programas más o menos integrados para hacer estas cosas, además de TestDesign, son los paquetes eatATA, ATA, xxIRT, dexterMST, catR, mstR, todos ellos en R, quizá el lenguaje más popular para esto. Algunos de estos paquetes están pensados para ensamblar exámenes adaptativos. En Julia, Python y SAS hay soluciones interesantes, aunque poco pulidas para el usuario, quien debe tener al menos un conocimiento básico de los lenguajes que dan forma a estos sistemas. En cierto modo, puede decirse que son librerías experimentales. Las grandes agencias de evaluación y aplicación generalmente desarrollan sus propias soluciones en casa.
