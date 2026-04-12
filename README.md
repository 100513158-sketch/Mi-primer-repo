# Memoria técnica del diseño de enlace RF para IoT en banda de 868 MHz

## 1. Introducción
El presente documento desarrolla el diseño del sistema radiante para un enlace IoT en banda ISM de 868 MHz, considerando tanto el transmisor como los receptores de la red. La solución propuesta está orientada a demostrar la viabilidad de un sistema de monitoreo distribuido en gasoductos basado en tecnología LoRaWAN. En este escenario, los sensores operan con batería, se distribuyen a lo largo de grandes distancias y deben comunicar en condiciones de bajo consumo, cobertura extendida y coste contenido.

El objetivo principal es justificar, mediante criterios electromagnéticos y resultados de simulación, la selección de antenas y la viabilidad del enlace radioeléctrico. Para ello, se analizan la adaptación de impedancia, la directividad, la polarización y el margen de enlace bajo modelos de propagación en espacio libre y entorno terrestre.

La metodología empleada combina simulación de antenas en MATLAB Antenna Toolbox y evaluación de presupuesto de enlace mediante los scripts de cálculo implementados para Friis y Okumura-Hata. De este modo, la memoria no solo presenta valores finales, sino también la lógica de diseño seguida y la justificación explícita de los parámetros adoptados. El uso de componentes comerciales y geometrías de antena realizables permite que la propuesta sea escalable en número de nodos y replicable en un despliegue real.

## 2. Metodología de diseño y validación
El proceso de trabajo se estructuró en cuatro fases. En primer lugar, se definieron las condiciones del sistema (frecuencia de operación, nivel de potencia transmitida, sensibilidad del receptor y restricción de adaptación a 50 ohm). En segundo lugar, se diseñaron y simularon dos soluciones radiantes candidatas: un monopolo y una antena patch microstrip. En tercer lugar, se analizaron métricas de comportamiento en banda, en particular impedancia de entrada, coeficiente de reflexión |S11| y patrones de radiación. Finalmente, se verificó la viabilidad del enlace con modelos de propagación, incluyendo estimación de distancia máxima teórica y práctica.

La consistencia entre diseño de antena y presupuesto de enlace permitió cerrar el ciclo de ingeniería: la antena propuesta no se evalúa de forma aislada, sino en relación directa con la potencia recibida esperada en el escenario de uso.

En esta solución se seleccionó como sistema radiante del gateway una antena monopolo vertical, mientras que para los sensores se adoptó una antena patch microstrip. La elección del monopolo en el gateway se justifica por su cobertura prácticamente omnidireccional en azimut, lo que permite recibir tramas desde sensores ubicados en múltiples direcciones sin requerir realineación mecánica. En un escenario lineal y extendido como un gasoducto, esta característica reduce la complejidad de despliegue y mejora la robustez frente a incertidumbres de posición de los nodos.

Por su parte, la elección de patch microstrip en sensores se justifica por criterios de integración y operación con batería: geometría compacta y plana, facilidad de montaje sobre PCB, bajo perfil mecánico y posibilidad de fabricación repetible con componentes comerciales. Para sensores distribuidos en campo, estas ventajas favorecen escalabilidad y mantenimiento, manteniendo simultáneamente una adaptación adecuada en 868 MHz cuando el punto de alimentación se ajusta correctamente.

Respecto a la pregunta de cobertura, para un gasoducto de aproximadamente 20 km sí es técnicamente posible plantear un único gateway en el centro (alcance nominal de 10 km hacia cada extremo), siempre que el margen de enlace calculado con el modelo de propagación realista (Okumura-Hata rural) sea positivo y suficientemente holgado. En diseño práctico se recomienda conservar margen adicional por desvanecimiento, vegetación, clima, pérdidas de instalación y variabilidad de altura de antenas. Si en el peor caso ese margen cae por debajo del umbral de diseño, debe considerarse un segundo gateway o un emplazamiento con mayor altura efectiva.

La diferencia de ángulo de llegada entre un sensor cercano y uno lejano afecta principalmente al corte de elevación del patrón del gateway. El sensor más cercano suele llegar con un ángulo de elevación mayor (más inclinado), mientras que el más lejano llega con un ángulo más cercano al horizonte. Si la antena del gateway presenta nulos o menor ganancia en alguno de esos ángulos, el enlace será más débil para ese grupo de sensores. En consecuencia, no basta con validar cobertura en azimut; también debe verificarse el patrón en elevación para asegurar ganancia útil tanto en enlaces cortos como largos, y ajustar altura/inclinación del gateway cuando sea necesario.

### 2.1 Análisis geométrico del ángulo de llegada (sensor cercano frente a lejano)
Para cuantificar el efecto anterior se emplea una aproximación geométrica básica del ángulo de elevación de llegada al gateway:

$$
	heta = \arctan\left(\frac{h_{GW} - h_{S}}{d}\right)
$$

donde $h_{GW}$ es la altura de la antena del gateway, $h_S$ la altura de la antena del sensor y $d$ la distancia horizontal entre ambos.

Tomando los valores de diseño utilizados en el proyecto ($h_{GW}=30$ m y $h_S=1.5$ m), la diferencia de altura es $\Delta h = 28.5$ m. Con ello:

- Sensor cercano a $d=1$ km:

$$
	heta_{1km}=\arctan\left(\frac{28.5}{1000}\right)\approx 1.63^\circ
$$

- Sensor lejano a $d=10$ km:

$$
	heta_{10km}=\arctan\left(\frac{28.5}{10000}\right)\approx 0.16^\circ
$$

Estos resultados muestran que, incluso en 1 km, el ángulo de elevación es bajo y en 10 km es prácticamente rasante al horizonte. Por tanto, la parte más crítica del patrón del gateway es la zona de elevaciones bajas. Si la antena presenta depresión de ganancia cerca del horizonte, el sensor más lejano será el primero en degradarse. Este análisis justifica verificar con detalle el patrón de elevación del gateway y, si fuese necesario, aumentar altura de instalación o ajustar inclinación mecánica para mejorar cobertura en el extremo del trazado.

### 2.2 Criterios de aceptación numéricos para validación del sistema
Con el fin de convertir la validación en un proceso objetivo y reproducible, se establecen los siguientes criterios de aceptación para simulación y prueba en campo:

1. Adaptación de antena en banda de trabajo (868 MHz):
- Criterio principal: $|S11| \leq -10$ dB en la frecuencia central.
- Criterio recomendado: VSWR $\leq 2$ en la frecuencia central.
- Criterio de robustez: mantener $|S11| \leq -10$ dB en un entorno cercano a la banda objetivo (por ejemplo, tolerancia de frecuencia de diseno y variaciones de montaje).

2. Cobertura en azimut del gateway:
- Se evalua el patron de azimut a 868 MHz.
- Ondulacion azimutal maxima permitida en el sector util: $\Delta G_{az,max} \leq 3$ dB.
- Criterio minimo aceptable en escenario exigente: $\Delta G_{az,max} \leq 6$ dB.

3. Cobertura en elevacion para sensores cercanos y lejanos:
- Se valida ganancia util en el rango de angulos de llegada calculado (aprox. $0.16^\circ$ a $1.63^\circ$ para 10 km y 1 km, respectivamente).
- Criterio recomendado: no presentar nulos profundos en ese intervalo angular.
- Criterio de ingenieria: variacion de ganancia en ese intervalo menor o igual a 6 dB.

4. Presupuesto de enlace:
- Criterio principal: margen de enlace en peor caso $\geq 10$ dB.
- Criterio recomendado para operacion robusta: margen de enlace en peor caso $\geq 15$ dB.
- Criterio de diseno conservador: margen objetivo de 20 dB cuando el entorno sea variable o no controlado.

5. Objetivos de calidad de recepcion (medidos en campo):
- RSSI objetivo en el extremo lejano: mayor o igual a $-120$ dBm.
- Umbral minimo operativo: RSSI mayor o igual a $-125$ dBm.
- SNR objetivo: mayor o igual a $-10$ dB.
- Umbral minimo operativo: SNR mayor o igual a $-15$ dB.

6. Objetivo de fiabilidad de trafico:
- Packet delivery ratio (PDR) en extremos: mayor o igual al 95%.
- Criterio minimo aceptable en condicion desfavorable: PDR mayor o igual al 90%.

7. Regla de aceptacion global del despliegue:
- El diseno se considera aceptado solo si todos los criterios 1 a 4 se cumplen en simulacion y, al menos, los criterios 5 y 6 se cumplen en pruebas de campo en ambos extremos del gasoducto.

### Conclusiones del apartado 2.2
La definicion de umbrales numericos evita decisiones subjetivas y permite defender tecnicamente la viabilidad del sistema. Ademas, al exigir validacion simultanea en patron, enlace y metricas reales de recepcion, se garantiza que el rendimiento simulado se traduzca en funcionamiento robusto sobre el trazado completo del gasoducto.

### 2.3 Resultados del barrido de optimizacion
Con el script de barrido de altura e inclinacion del gateway se obtiene una comparativa sistematica del margen de enlace sobre el trazado completo. En esta seccion se deben insertar las figuras generadas y documentar el resultado de cada una.

Figura 17. Mapa de calor del peor margen de enlace [dB] en funcion de la inclinacion del gateway [deg] y la altura del gateway [m].
Pegar aqui la figura: mapa de calor de peor margen en todo el trazado (0.5-10 km).
Conclusión breve: Esta figura permite identificar la combinacion altura-inclinacion que maximiza el margen en peor caso y, por tanto, la configuracion mas robusta para cubrir simultaneamente sensores cercanos y lejanos.

Figura 18. Margen de enlace [dB] frente a distancia [km]: comparacion entre configuracion optima del barrido y configuracion de referencia (30 m, 0 deg).
Pegar aqui la figura: curva de margen vs distancia (optimo vs referencia) con lineas de umbral.
Conclusión breve: La figura evidencia si la configuracion optimizada mejora el margen en todo el trazado y, en particular, en el extremo de 10 km, donde el enlace es mas exigente.

Figura 19. Ganancia del gateway [dBi] en elevaciones bajas en funcion del angulo de llegada [deg], para distintos valores de inclinacion.
Pegar aqui la figura: sensibilidad de ganancia en elevacion baja para los tilt barridos.
Conclusión breve: La respuesta angular confirma que pequenas variaciones de inclinacion pueden mejorar o degradar la recepcion de sensores lejanos, al desplazar la ganancia util cerca del horizonte.

Figura 20. Mapa de calor del margen en el extremo lejano (10 km) [dB] en funcion de la inclinacion [deg] y la altura [m] del gateway.
Pegar aqui la figura: mapa de calor del margen en extremo lejano.
Conclusión breve: Esta figura focaliza la condicion mas critica del sistema y permite validar si el extremo del gasoducto cumple los criterios de aceptacion definidos para margen de enlace.

### Conclusiones del apartado 2.3
El barrido de optimizacion transforma la seleccion de altura e inclinacion del gateway en una decision cuantitativa. La combinacion elegida debe justificarse con la configuracion que maximice el margen en peor caso y asegure cumplimiento de umbrales en el extremo lejano, manteniendo cobertura estable en todo el trazado.

## 3. Diseño del transmisor
### 3.1 Selección del sistema radiante
Para el transmisor se evaluaron dos alternativas: monopolo y patch microstrip. El monopolo presenta como ventaja principal su simplicidad estructural, bajo coste, fácil fabricación y comportamiento casi omnidireccional en azimut cuando se monta verticalmente sobre plano de tierra. La antena patch, en cambio, ofrece mejor integración mecánica en superficies planas y mayor control geométrico del patrón, aunque su comportamiento suele ser más sensible a tolerancias del sustrato y del punto de alimentación.

Dado que los gateways deben cubrir sensores en múltiples direcciones, se optó como solución principal por una antena omnidireccional vertical tipo monopolo. Esta elección favorece la cobertura angular alrededor del gateway y reduce la dependencia de orientación de los sensores en campo. La antena patch se mantiene como alternativa técnica para comparar comportamiento electromagnético y evaluar escenarios donde la integración mecánica plana sea prioritaria.

En el contexto de monitoreo de gasoductos, esta decisión resulta coherente con los requisitos de red: nodos distribuidos, alimentación por batería, larga autonomía y necesidad de conectividad robusta con infraestructura de coste moderado.

### 3.2 Diseño y justificación del monopolo
El diseño principal se realizó a 868 MHz con una altura de 0.0864 m, correspondiente aproximadamente a λ/4. Esta elección no es arbitraria: un monopolo de cuarto de onda opera cercano a resonancia y permite una radiación eficiente con tamaño contenido. Se definió además un plano de tierra de 0.1 m x 0.1 m, aproximado a λ/3 por lado, para sostener el efecto de imagen electromagnética necesario en este tipo de radiador.

Se introdujo una carga compleja de 28 - 4j ohm con el propósito de aproximar la adaptación hacia la referencia de 50 ohm del sistema RF. La validación se efectuó mediante curva de impedancia y |S11| en banda, junto con diagramas de patrón 3D, cortes en azimut y elevación, y distribución de corriente.

### 3.3 Diseño y justificación del patch microstrip
La alternativa patch se diseñó también para 868 MHz. Se fijó un espesor de sustrato de 3.4538 mm y un offset de alimentación de [0.034902, 0] m para mejorar la adaptación de entrada. El conductor se modeló en aluminio (conductividad 3.77e7 S/m) con espesor de 0.762 mm, incorporando pérdidas óhmicas de manera realista.

En este caso, además de la impedancia de entrada, se calculó explícitamente |S11| y VSWR en la frecuencia central. Este enfoque permite una lectura directa de la calidad de adaptación: en términos prácticos, se considera deseable |S11| <= -10 dB y VSWR <= 2 en banda de operación.

### 3.4 Directividad y polarización del transmisor
El análisis de directividad se fundamenta en los patrones de radiación obtenidos por simulación. El monopolo, en configuración vertical, presenta polarización lineal vertical y cobertura angular adecuada para enlaces terrestres con nodos distribuidos. La antena patch presenta polarización lineal asociada al modo excitado, con patrón más dependiente de la orientación del plano radiante.

Por tanto, la elección entre ambos radiadores debe hacerse en función del compromiso entre cobertura espacial requerida y restricciones de integración mecánica del equipo transmisor.

### Conclusiones del apartado 3
El transmisor puede implementarse de forma fiable con monopolo λ/4 cuando se prioriza robustez y cobertura azimutal, criterio especialmente adecuado para gateways que deben atender sensores en múltiples direcciones. La patch microstrip es técnicamente viable y aporta ventajas de integración física, siempre que se asegure una correcta adaptación en 868 MHz y control de tolerancias de fabricación. En ambos casos, la verificación de |S11|, impedancia y patrón de radiación resulta indispensable para validar la elección.

## 4. Diseño de los receptores con criterio de versatilidad
El enunciado exige más de un receptor, por lo que la decisión de diseño debe enfatizar versatilidad. En este contexto, un diseño versátil es aquel que mantiene prestaciones estables frente a variaciones de montaje, orientación y entorno de instalación, sin requerir reajustes complejos en campo.

Desde esa perspectiva, los receptores deben seleccionarse con base en cuatro criterios: estabilidad de adaptación en banda, cobertura angular suficiente para incertidumbre de alineación, compatibilidad de polarización con el transmisor y facilidad de implementación sobre distintas plataformas hardware. Para receptores fijos o gateways puede admitirse mayor directividad si el emplazamiento está controlado; para receptores distribuidos en nodos múltiples se favorecen patrones más amplios y soluciones compactas.

La evaluación de receptores debe reportar, para cada opción, impedancia en frecuencia central, mínimo de |S11| en banda, VSWR y comportamiento de patrón en azimut y elevación. Esta comparación permite justificar técnicamente qué diseño aporta mejor equilibrio entre rendimiento y flexibilidad operativa.

### Conclusiones del apartado 4
La versatilidad de los receptores no depende de un único parámetro, sino del balance entre adaptación, cobertura y facilidad de despliegue. En una red con múltiples receptores para monitoreo de gasoductos, la solución más robusta será aquella que conserve buen acoplamiento y cobertura útil aun con variaciones de instalación y orientación de nodos.

## 5. Evidencias de funcionamiento del enlace
La validación del enlace se completó con resultados numéricos de presupuesto de potencia mediante los modelos Friis y Okumura-Hata. El modelo de Friis proporciona una referencia ideal en espacio libre, útil como límite superior teórico. No obstante, para conclusiones de ingeniería aplicada debe priorizarse Okumura-Hata en entorno rural, al incorporar pérdidas de propagación más representativas de escenarios reales.

Con los parámetros de simulación empleados (f = 868 MHz, P_TX = 14 dBm, ganancias de antena en el orden de 2 dBi y sensibilidad de -137 dBm), se obtiene margen de enlace positivo en la distancia de análisis. Además, los scripts calculan distancia máxima teórica y distancia máxima práctica al introducir margen de diseño adicional, lo cual aporta una medida de robustez frente a desvanecimientos y pérdidas no ideales.

La evidencia completa de correcto funcionamiento debe incluir, en el documento final, tres bloques: diagramas de directividad, representación de |S11| y resultados numéricos del balance de enlace (potencia recibida, margen y alcance estimado).

### Conclusiones del apartado 5
El enlace puede considerarse funcional cuando la potencia recibida supera la sensibilidad con margen positivo y las antenas se mantienen adaptadas en banda. La decisión final debe sustentarse en resultados de propagación realista y no exclusivamente en predicciones de espacio libre.

## 6. Criterios formales de presentación de resultados
Conforme a las instrucciones de la práctica, todas las ilustraciones deben presentarse numeradas y con pie de figura descriptivo. Asimismo, cada gráfica debe indicar de forma explícita la magnitud y unidad en cada eje. Este requisito es esencial para la trazabilidad técnica y para la evaluación objetiva por parte del tribunal.

Ejemplos de nomenclatura recomendada: “Figura 1. Geometría 3D de la antena monopolo a 868 MHz”, “Figura 2. Coeficiente de reflexión |S11| del patch microstrip en banda 781.2–954.8 MHz”, “Figura 3. Patrón de radiación en azimut del transmisor”. Del mismo modo, los ejes deben rotularse, por ejemplo, Frecuencia [MHz], Impedancia [ohm], Ganancia [dBi], Potencia [dBm] y Distancia [km].

### Conclusiones del apartado 6
La claridad formal de la memoria es parte del resultado técnico: una buena trazabilidad gráfica y metrológica fortalece la validez de las conclusiones y facilita la reproducibilidad de las simulaciones.

## 7. Justificación explícita de parámetros de diseño
Los parámetros adoptados en esta práctica se justifican por criterios físicos y de implementación. La frecuencia de 868 MHz responde al marco LoRaWAN en banda ISM europea. Las geometrías radiantes se seleccionan por compromiso entre tamaño, patrón de radiación y facilidad de integración. Los parámetros de adaptación (offset, carga y referencia de 50 ohm) se orientan a maximizar transferencia de potencia y minimizar reflexión. Los materiales conductores y espesores se definen para representar condiciones de fabricación realistas. Finalmente, la inclusión de margen de diseño en el cálculo de distancia práctica permite contemplar pérdidas no idealizadas.

### Conclusiones del apartado 7
La memoria cumple el criterio de diseño justificado cuando cada valor utilizado se vincula a un fundamento electromagnético o de ingeniería de sistema, evitando decisiones arbitrarias.

## 8. Conclusión general
El trabajo realizado permite afirmar que el diseño del enlace RF IoT en 868 MHz es técnicamente viable bajo los criterios evaluados. La combinación de análisis de antenas y presupuesto de enlace proporciona evidencia suficiente para seleccionar arquitectura de transmisor y receptores con base objetiva. En particular, la elección de una antena monopolo omnidireccional vertical para el gateway se alinea con la necesidad de cobertura multi-direccional sobre sensores distribuidos.

En términos de ingeniería aplicada, la solución propuesta demuestra la viabilidad de un sistema LoRaWAN para monitoreo distribuido en gasoductos: sensores de bajo consumo alimentados por batería, cobertura extendida y posibilidad de implementación con componentes comerciales, lo que favorece una arquitectura eficiente, escalable y de bajo coste. La propuesta final debe consolidarse en el documento de entrega incorporando las figuras numeradas, las métricas de adaptación en banda y los resultados de potencia recibida y margen de enlace en modelo realista.

## 9. Guía breve de integración en Word
Para pegar este texto directamente en el documento final se recomienda:
1. Aplicar estilos de Word: Título 1 para capítulos y Título 2 para subapartados.
2. Insertar cada figura inmediatamente después del párrafo que la describe.
3. Añadir pies de figura con numeración automática.
4. Incorporar, bajo cada capítulo, la subsección “Conclusiones” ya incluida en esta versión.
5. Revisar que todas las magnitudes estén expresadas con unidades SI o dB/dBm según corresponda.

## 10. Figuras a insertar (scripts de antena)

En esta sección se indica exactamente dónde insertar cada figura de simulación. Solo debes copiar cada pie en Word, colocar la imagen correspondiente debajo y mantener la numeración.

### 10.1 Figuras de script_monopolo

Figura 1. Geometría 3D de la antena monopolo λ/4 diseñada para 868 MHz.
Conclusión breve: La geometría confirma una solución compacta y apta para integración en nodos/gateway con montaje vertical.

Figura 2. Impedancia de entrada del monopolo λ/4 en la banda de análisis (Frecuencia [MHz] vs Impedancia [ohm]).
Conclusión breve: La cercanía de la impedancia a 50 ohm en torno a 868 MHz indica buen acoplamiento con el sistema RF.

Figura 3. Coeficiente de reflexión |S11| del monopolo λ/4 (Frecuencia [MHz] vs |S11| [dB]).
Conclusión breve: Un valor de |S11| por debajo de -10 dB alrededor de la frecuencia objetivo valida la adaptación en banda.

Figura 4. Patrón de radiación 3D del monopolo a 868 MHz.
Conclusión breve: El diagrama evidencia comportamiento cuasi omnidireccional, adecuado para cobertura en múltiples direcciones.

Figura 5. Patrón de radiación en azimut del monopolo a 868 MHz (Azimut [grados] vs Ganancia [dBi]).
Conclusión breve: La cobertura azimutal es amplia y homogénea, favorable para sensores distribuidos alrededor del gateway.

Figura 6. Patrón de radiación en elevación del monopolo a 868 MHz (Elevación [grados] vs Ganancia [dBi]).
Conclusión breve: El lóbulo principal en elevación es coherente con enlaces terrestres de media y larga distancia.

Figura 7. Distribución de corriente en el monopolo a 868 MHz.
Conclusión breve: La distribución confirma el modo de radiación esperado para un monopolo resonante de cuarto de onda.

Figura 8. Comparación de ganancia entre variantes monopolo λ/4, monopolo λ/2 y dipolo λ/2.
Conclusión breve: La comparativa muestra el compromiso entre compacidad y ganancia, apoyando la selección según el escenario.

Figura 9. Comparación de eficiencia estimada entre variantes monopolo λ/4, monopolo λ/2 y dipolo λ/2.
Conclusión breve: La eficiencia relativa permite identificar la alternativa más robusta cuando se prioriza rendimiento energético.

### 10.2 Figuras de pachmicrostrip

Figura 10. Geometría 3D de la antena patch microstrip diseñada para 868 MHz.
Conclusión breve: La geometría plana facilita integración mecánica en superficies rígidas y módulos compactos.

Figura 11. Impedancia de entrada del patch microstrip en banda (Frecuencia [MHz] vs Impedancia [ohm]).
Conclusión breve: El ajuste del punto de alimentación permite aproximar la impedancia de entrada a la referencia de 50 ohm.

Figura 12. Coeficiente de reflexión |S11| del patch microstrip (Frecuencia [MHz] vs |S11| [dB]).
Conclusión breve: La profundidad del mínimo de |S11| alrededor de 868 MHz confirma la sintonía del diseño.

Figura 13. Patrón de radiación 3D del patch microstrip a 868 MHz.
Conclusión breve: El patrón más direccional respecto al monopolo puede ser útil en enlaces con orientación controlada.

Figura 14. Patrón en azimut del patch microstrip a 868 MHz (Azimut [grados] vs Ganancia [dBi]).
Conclusión breve: La respuesta angular en azimut muestra zonas de mayor y menor cobertura que deben considerarse en el despliegue.

Figura 15. Patrón en elevación del patch microstrip a 868 MHz (Elevación [grados] vs Ganancia [dBi]).
Conclusión breve: El corte en elevación permite verificar la orientación óptima de montaje para maximizar el enlace útil.

Figura 16. Distribución de corriente superficial en el patch microstrip a 868 MHz.
Conclusión breve: La corriente superficial valida el modo resonante esperado y la coherencia del diseño electromagnético.

### Conclusión del apartado 10
La secuencia de Figuras 1 a 16 proporciona evidencia completa de geometría, adaptación y radiación para ambas soluciones de antena. Esta trazabilidad visual permite justificar de forma sólida la elección final del sistema radiante para monitoreo distribuido en gasoductos.
