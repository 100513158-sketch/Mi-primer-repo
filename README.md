# Memoria técnica del diseño de enlace RF para IoT en banda de 868 MHz

## 1. Introducción
El presente documento desarrolla el diseño del sistema radiante para un enlace IoT en banda ISM de 868 MHz, considerando tanto el transmisor como los receptores de la red. La solución propuesta está orientada a demostrar la viabilidad de un sistema de monitoreo distribuido en gasoductos basado en tecnología LoRaWAN. En este escenario, los sensores operan con batería, se distribuyen a lo largo de grandes distancias y deben comunicar en condiciones de bajo consumo, cobertura extendida y coste contenido.

El objetivo principal es justificar, mediante criterios electromagnéticos y resultados de simulación, la selección de antenas y la viabilidad del enlace radioeléctrico. Para ello, se analizan la adaptación de impedancia, la directividad, la polarización y el margen de enlace bajo modelos de propagación en espacio libre y entorno terrestre.

La metodología empleada combina simulación de antenas en MATLAB Antenna Toolbox y evaluación de presupuesto de enlace mediante los scripts de cálculo implementados para Friis y Okumura-Hata. De este modo, la memoria no solo presenta valores finales, sino también la lógica de diseño seguida y la justificación explícita de los parámetros adoptados. El uso de componentes comerciales y geometrías de antena realizables permite que la propuesta sea escalable en número de nodos y replicable en un despliegue real.

## 2. Metodología de diseño y validación
El proceso de trabajo se estructuró en cuatro fases. En primer lugar, se definieron las condiciones del sistema (frecuencia de operación, nivel de potencia transmitida, sensibilidad del receptor y restricción de adaptación a 50 ohm). En segundo lugar, se estableció la arquitectura final del sistema radiante: antena monopolo en el gateway y antena patch microstrip en los sensores fijos desplegados a lo largo de 20 km. En tercer lugar, se analizaron métricas de comportamiento en banda, en particular impedancia de entrada, coeficiente de reflexión |S11|, patrones de radiación, polarización y sensibilidad a ángulos de llegada. Finalmente, se verificó la viabilidad del enlace con modelos de propagación, incluyendo estimación de distancia máxima teórica y práctica.

La consistencia entre diseño de antena y presupuesto de enlace permitió cerrar el ciclo de ingeniería: la antena propuesta no se evalúa de forma aislada, sino en relación directa con la potencia recibida esperada en el escenario de uso.

En esta solución se seleccionó como sistema radiante del gateway una antena monopolo vertical, mientras que para los sensores se adoptó una antena patch microstrip. La elección del monopolo en el gateway se justifica por su cobertura prácticamente omnidireccional en azimut, lo que permite recibir tramas desde sensores ubicados en múltiples direcciones sin requerir realineación mecánica. En un escenario lineal y extendido como un gasoducto, esta característica reduce la complejidad de despliegue y mejora la robustez frente a incertidumbres de posición de los nodos.

Por su parte, la elección de patch microstrip en sensores se justifica por criterios de integración y operación con batería: geometría compacta y plana, facilidad de montaje sobre PCB, bajo perfil mecánico y posibilidad de fabricación repetible con componentes comerciales. Adicionalmente, al tratarse de sensores fijos, su directividad permite orientar el lóbulo principal hacia el gateway para concentrar energía en la dirección de interés y mejorar la ganancia efectiva del enlace sensor-gateway. Para sensores distribuidos en campo, estas ventajas favorecen escalabilidad y mantenimiento, manteniendo simultáneamente una adaptación adecuada en 868 MHz cuando el punto de alimentación se ajusta correctamente.

De forma crítica, el radioenlace requiere compatibilidad de polarización entre ambas antenas. Por tanto, la orientación mecánica del monopolo en gateway y de los patches en sensores debe fijarse de forma consistente durante la instalación para minimizar pérdidas por desacoplo de polarización.

Con los resultados obtenidos en esta memoria, se confirma que para un gasoducto de aproximadamente 20 km es viable operar con un único gateway central (10 km por lado). Al aplicar un criterio de diseño orientado a coste de infraestructura y robustez en el extremo, la altura mínima robusta en 10 km resulta 9 m con tilt -1 deg y margen de 15.39 dB (L_pol=0 dB, umbral de robustez 15 dB). Como referencia de máxima robustez en el rango barrido, la mejor configuración por peor caso es 50 m con tilt 0 deg y margen de 30.56 dB. Por criterio de ingeniería, se mantiene la recomendación de reservar margen adicional frente a desvanecimiento, vegetación, clima y pérdidas de instalación; si en validación de campo el peor caso descendiera por debajo del umbral adoptado, se reevaluaría altura/emplazamiento o se incorporaría un segundo gateway.

El análisis geométrico confirma que los ángulos de llegada en elevación son bajos en todo el trazado (aprox. 0.49 deg a 1 km y 0.05 deg a 10 km para $h_{GW}=10$ m y $h_S=1.5$ m). En consecuencia, la cobertura efectiva queda gobernada por la respuesta del gateway cerca del horizonte: cualquier depresión de ganancia en elevaciones bajas penaliza primero el extremo lejano. Por ello, además de validar azimut, se debe verificar explícitamente el patrón de elevación en ese intervalo angular y ajustar altura/tilt del gateway si se detectan degradaciones en el peor caso.

### 2.1 Análisis geométrico del ángulo de llegada (sensor cercano frente a lejano)
Para cuantificar el efecto anterior se emplea una aproximación geométrica básica del ángulo de elevación de llegada al gateway:

$$
θ = \arctan\left(\frac{h_{GW} - h_{S}}{d}\right)
$$

donde $h_{GW}$ es la altura de la antena del gateway, $h_S$ la altura de la antena del sensor y $d$ la distancia horizontal entre ambos.

Tomando los valores de diseño utilizados en el proyecto ($h_{GW}=10$ m y $h_S=1.5$ m), la diferencia de altura es $\Delta h = 8.5$ m. Con ello:

- Sensor cercano a $d=1$ km:

$$
θ_{1km}=\arctan\left(\frac{8.5}{1000}\right)\approx 0.49^\circ
$$

- Sensor lejano a $d=10$ km:

$$
θ_{10km}=\arctan\left(\frac{8.5}{10000}\right)\approx 0.05^\circ
$$

Por tanto, la parte más crítica del patrón del gateway es la zona de elevaciones bajas. Si la antena presenta depresión de ganancia cerca del horizonte, el sensor más lejano será el primero en degradarse. Este análisis justifica verificar con detalle el patrón de elevación del gateway y, si fuese necesario, aumentar altura de instalación o ajustar inclinación mecánica para mejorar cobertura en el extremo del trazado.

### 2.2 Criterios de aceptación numéricos para validación del sistema
Con el fin de convertir la validación en un proceso objetivo y reproducible, se establecen los siguientes criterios de aceptación para simulación y prueba en campo:

1. Adaptación de antena en banda de trabajo (868 MHz):
- Criterio principal: $|S11| \leq -10$ dB en la frecuencia central.
- Criterio recomendado: VSWR $\leq 2$ en la frecuencia central.
- Criterio de robustez: mantener $|S11| \leq -10$ dB en un entorno cercano a la banda objetivo (por ejemplo, tolerancia de frecuencia de diseño y variaciones de montaje).

2. Cobertura en azimut del gateway:
- Se evalúa el patrón de azimut a 868 MHz.
- Ondulación azimutal máxima permitida en el sector útil: $\Delta G_{az,max} \leq 3$ dB.
- Criterio mínimo aceptable en escenario exigente: $\Delta G_{az,max} \leq 6$ dB.

3. Cobertura en elevación para sensores cercanos y lejanos:
- Se valida ganancia útil en el rango de ángulos de llegada calculado (aprox. $0.16^\circ$ a $1.63^\circ$ para 10 km y 1 km, respectivamente).
- Criterio recomendado: no presentar nulos profundos en ese intervalo angular.
- Criterio de ingeniería: variación de ganancia en ese intervalo menor o igual a 6 dB.

4. Presupuesto de enlace:
- Criterio principal: margen de enlace en peor caso $\geq 10$ dB.
- Criterio recomendado para operación robusta: margen de enlace en peor caso $\geq 15$ dB.
- Criterio de diseño conservador: margen objetivo de 10 dB cuando el entorno sea variable o no controlado.

5. Objetivos de calidad de recepción (medidos en campo):
- RSSI objetivo en el extremo lejano: mayor o igual a $-120$ dBm.
- Umbral mínimo operativo: RSSI mayor o igual a $-125$ dBm.
- SNR objetivo: mayor o igual a $-10$ dB.
- Umbral mínimo operativo: SNR mayor o igual a $-15$ dB.

6. Objetivo de fiabilidad de tráfico:
- Packet delivery ratio (PDR) en extremos: mayor o igual al 95%.
- Criterio mínimo aceptable en condición desfavorable: PDR mayor o igual al 90%.

7. Regla de aceptación global del despliegue:
- El diseño se considera aceptado solo si todos los criterios 1 a 4 se cumplen en simulación y, al menos, los criterios 5 y 6 se cumplen en pruebas de campo en ambos extremos del gasoducto.

### Conclusiones del apartado 2.2
La definición de umbrales numéricos evita decisiones subjetivas y permite defender técnicamente la viabilidad del sistema. Además, al exigir validación simultánea de patrón, enlace y métricas reales de recepción, se garantiza que el rendimiento simulado se traduzca en funcionamiento robusto sobre el trazado completo del gasoducto.

### 2.3 Resultados del barrido de optimización
Con el script de barrido de altura e inclinación del gateway se obtiene una comparativa sistemática del margen de enlace sobre el trazado completo. En esta sección se insertan las figuras generadas y se documenta el resultado de cada una.

Figura 17. Mapa de calor del peor margen de enlace [dB] en función de la inclinación del gateway [deg] y la altura del gateway [m], junto con la curva de margen vs distancia para la configuración óptima.
Insertar aquí la figura correspondiente: panel combinado con mapa de calor, curva de margen vs distancia y etiquetas de distancia (por ejemplo 0.5 km, 1 km, 5 km y 10 km).
Conclusión breve: Esta figura permite identificar la combinación altura-inclinación que maximiza el margen mínimo de enlace y, además, visualizar de forma intuitiva cómo evoluciona la compatibilidad del radioenlace sensor-gateway a lo largo del trazado.

Figura 18. Margen de enlace [dB] frente a distancia [km]: comparación entre configuración óptima del barrido y configuración mínima robusta.
Insertar aquí la figura correspondiente: curva de margen vs distancia (óptimo vs referencia) con líneas de umbral.
Conclusión breve: La figura evidencia si la configuración optimizada mejora la compatibilidad del enlace en toda la línea y, en particular, en el extremo de 10 km, donde la exigencia de ganancia y polarización es mayor.

Figura 19. Ganancia del gateway [dBi] en elevaciones bajas en función del ángulo de llegada [deg], para distintos valores de inclinación.
Insertar aquí la figura correspondiente: sensibilidad de ganancia en elevación baja para los tilt barridos.
Conclusión breve: La respuesta angular confirma cómo pequeñas variaciones de inclinación alteran la compatibilidad angular del radioenlace, pudiendo mejorar o degradar la recepción de sensores lejanos.

Figura 20. Mapa de calor del margen en el extremo lejano (10 km) [dB] en función de la inclinación [deg] y la altura [m] del gateway.
Insertar aquí la figura correspondiente: mapa de calor del margen en extremo lejano.
Conclusión breve: Esta figura focaliza la condición más crítica del sistema y valida si la configuración mantiene compatibilidad de antenas y margen suficiente en el caso de peor distancia.

### Conclusiones del apartado 2.3
El barrido de optimización transforma la selección de altura e inclinación del gateway en una decisión cuantitativa. La combinación elegida debe justificarse con la configuración que maximice el margen en peor caso y asegure cumplimiento de umbrales en el extremo lejano, manteniendo compatibilidad de polarización y cobertura angular estable en todo el trazado.

Resultados obtenidos en la ejecución del barrido (caso $L_{pol}=0$ dB):
- Ganancia efectiva del sensor patch (broadside): 9.76 dBi.
- Criterio de robustez en 10 km: margen >= 15 dB.
- Altura mínima robusta: 9 m, tilt -1 deg, margen en 10 km = 15.39 dB.
- Mejor configuración por peor caso (referencia de robustez máxima): altura de gateway 50 m, tilt 0 deg.
- Margen peor caso (0.5-10 km) en configuración robusta máxima: 30.56 dB.
- Margen en extremo lejano (10 km) en configuración robusta máxima: 30.56 dB.
- Configuración de comparación (10 m, 0 deg): margen en 10 km = 26.04 dB.

Interpretación: con los parámetros actuales, el radioenlace es viable en todo el trazado de 20 km incluso con una solución de menor coste estructural (9 m). Si se prioriza robustez máxima dentro del rango barrido, 50 m aporta margen adicional. En ambos enfoques se cumple el umbral de 15 dB en 10 km, con una diferencia de margen significativa a favor de la configuración más alta.

### 2.4 Comparativa por desacoplo de polarización: casos $L_{pol}=0$ dB y $L_{pol}=3$ dB
Para analizar la sensibilidad del radioenlace frente a desalineación de polarización, se comparan dos escenarios:

- Caso A (referencia): $L_{pol}=0$ dB (polarización alineada).
- Caso B (desacoplo moderado): $L_{pol}=3$ dB.

En el modelo implementado en el script de barrido, la pérdida por polarización se resta directamente en la potencia recibida:

$$
P_R = P_T + G_T + G_R - L_{rural} - L_{pol}
$$

Por tanto, al pasar de $L_{pol}=0$ dB a $L_{pol}=3$ dB, el margen de enlace se reduce exactamente en 3 dB para todas las distancias y configuraciones:

$$
M_{3dB}(d)=M_{0dB}(d)-3
$$

Implicaciones prácticas:

- La configuración óptima de altura/inclinación no debería cambiar por este barrido de polarización (el desplazamiento es uniforme).
- El margen en peor caso y el margen en el extremo de 10 km disminuyen 3 dB.
- La condición de aceptación debe revalidarse con el umbral exigido (10 dB o 15 dB según criterio de diseño).

Resultados comparativos obtenidos:
- Caso $L_{pol}=0$ dB (configuración robusta de referencia): margen peor caso = 30.56 dB; margen a 10 km = 30.56 dB.
- Caso $L_{pol}=3$ dB (inferencia del modelo): margen peor caso = 27.56 dB; margen a 10 km = 27.56 dB.
- Configuración robusta en ambos casos: altura de gateway = 50 m; tilt = 0 deg.

Redacción comparativa propuesta:

"Al comparar los casos $L_{pol}=0$ dB y $L_{pol}=3$ dB, se observa una degradación uniforme de 3 dB en el margen de enlace a lo largo de todo el trazado. La configuración óptima de altura e inclinación del gateway se mantiene, pero el margen en peor caso y en el extremo lejano disminuyen en la misma cuantía. En consecuencia, la compatibilidad de polarización entre monopolo (gateway) y patch (sensores) resulta crítica para preservar el cumplimiento de los criterios de aceptación del enlace." 

### Conclusiones del apartado 2.4
El estudio confirma que una desalineación moderada de polarización impacta directamente la robustez del radioenlace. Mantener polarización compatible y orientación mecánica consistente entre gateway y sensores es una condición de diseño esencial para asegurar cobertura de extremo a extremo en el gasoducto.

## 3. Diseño del sistema radiante del enlace
### 3.1 Selección del sistema radiante
En la solución final del proyecto no se emplea una única antena para todos los nodos, sino una arquitectura mixta: monopolo en el gateway y patch microstrip en los sensores. El monopolo presenta como ventaja principal su simplicidad estructural, bajo coste, fácil fabricación y comportamiento casi omnidireccional en azimut cuando se monta verticalmente sobre plano de tierra. La antena patch, en cambio, ofrece mejor integración mecánica en superficies planas y mayor control geométrico del patrón, lo que resulta especialmente útil en sensores fijos orientables.

Dado que los gateways deben cubrir sensores en múltiples direcciones, se optó por una antena omnidireccional vertical tipo monopolo en el gateway. Para los sensores desplegados a lo largo del gasoducto se seleccionó patch microstrip, aprovechando su directividad para orientar cada sensor hacia el gateway y aumentar la ganancia efectiva en la dirección del enlace.

En el contexto de monitoreo de gasoductos, esta decisión resulta coherente con los requisitos de red: nodos distribuidos, alimentación por batería, larga autonomía y necesidad de conectividad robusta con infraestructura de coste moderado.

### 3.2 Diseño y justificación del monopolo
El diseño principal se realizó a 868 MHz con una altura de 0.0864 m, correspondiente aproximadamente a λ/4. Esta elección no es arbitraria: un monopolo de cuarto de onda opera cercano a resonancia y permite una radiación eficiente con tamaño contenido. Se definió además un plano de tierra de 0.1 m x 0.1 m, aproximado a λ/3 por lado, para sostener el efecto de imagen electromagnética necesario en este tipo de radiador.

Se introdujo una carga compleja de 28 - 4j ohm con el propósito de aproximar la adaptación hacia la referencia de 50 ohm del sistema RF. La validación se efectuó mediante curva de impedancia y |S11| en banda, junto con diagramas de patrón 3D, cortes en azimut y elevación, y distribución de corriente.

Resultado numérico en 868 MHz (script de monopolo):
- Impedancia de entrada: $Z_{in}=50.33-j1.07$ ohm.
- Magnitud de impedancia: $|Z_{in}|=50.34$ ohm.

Estos valores confirman una excelente adaptación del monopolo al sistema de 50 ohm del gateway.

### 3.3 Diseño y justificación del patch microstrip
La alternativa patch se diseñó también para 868 MHz. Se fijó un espesor de sustrato de 3.4538 mm y un offset de alimentación de [0.034902, 0] m para mejorar la adaptación de entrada. El conductor se modeló en aluminio (conductividad 3.77e7 S/m) con espesor de 0.762 mm, incorporando pérdidas óhmicas de manera realista.

En este caso, además de la impedancia de entrada, se calculó explícitamente |S11| y VSWR en la frecuencia central. Este enfoque permite una lectura directa de la calidad de adaptación: en términos prácticos, se considera deseable |S11| <= -10 dB y VSWR <= 2 en banda de operación.

Resultados numéricos en 868 MHz (script de patch):
- Impedancia de entrada: $Z_{in}=51.26+j3.44$ ohm.
- $S11=-43.41$ dB.
- VSWR = 1.01.

Estos resultados muestran una adaptación excelente, plenamente compatible con el radioenlace propuesto.

### 3.4 Directividad, polarización y compatibilidad del radioenlace
El análisis de directividad se fundamenta en los patrones de radiación obtenidos por simulación. El monopolo, en configuración vertical, presenta polarización lineal vertical y cobertura angular adecuada para enlaces terrestres con nodos distribuidos. La antena patch presenta polarización lineal asociada al modo excitado, con patrón más dependiente de la orientación del plano radiante.

Por tanto, el correcto funcionamiento del enlace depende de cuatro factores simultáneos: cobertura azimutal del monopolo del gateway, orientación directiva de cada patch de sensor hacia el gateway, compatibilidad de polarización entre ambos extremos y ganancia útil para los distintos ángulos de entrada de sensores cercanos y lejanos.

### Conclusiones del apartado 3
La arquitectura final del sistema radiante queda definida por monopolo en el gateway y patch en sensores fijos. Esta combinación permite cobertura multiazimut en el gateway y ganancia directiva en cada sensor orientado al centro de red. 

En la comparación de variantes del monopolo, la opción λ/2 presenta mayor ganancia teórica, pero la opción λ/4 se adopta por mejor compromiso entre compactación mecánica, simplicidad de integración y adaptación eléctrica confirmada a 50 ohm en el gateway. Para el sensor, el patch microstrip se adopta por su directividad orientable y su excelente adaptación en banda (S11 y VSWR).

Nota metodológica: en la tabla comparativa automática del script de monopolo, los valores de eficiencia mostrados son indicadores internos de comparación y no deben interpretarse como eficiencias físicas absolutas. Para la toma de decisión se priorizan ganancia, adaptación, patrón y compatibilidad de instalación.

La viabilidad final exige verificar simultáneamente adaptación en banda, polarización compatible y respuesta angular frente a los ángulos de entrada esperados a lo largo de los 20 km.

## 4. Diseño de los receptores con criterio de versatilidad
El enunciado exige más de un receptor, por lo que la decisión de diseño debe enfatizar versatilidad. En este contexto, un diseño versátil es aquel que mantiene prestaciones estables frente a variaciones de montaje, orientación y entorno de instalación, sin requerir reajustes complejos en campo.

Desde esa perspectiva, los receptores deben seleccionarse con base en cuatro criterios: estabilidad de adaptación en banda, cobertura angular suficiente para incertidumbre de alineación, compatibilidad de polarización con el transmisor y facilidad de implementación sobre distintas plataformas hardware. Para receptores fijos o gateways puede admitirse mayor directividad si el emplazamiento está controlado; para receptores distribuidos en nodos múltiples se favorecen patrones más amplios y soluciones compactas.

La evaluación de receptores debe reportar, para cada opción, impedancia en frecuencia central, mínimo de |S11| en banda, VSWR y comportamiento de patrón en azimut y elevación. Esta comparación permite justificar técnicamente qué diseño aporta mejor equilibrio entre rendimiento y flexibilidad operativa.

### Conclusiones del apartado 4
La versatilidad de los receptores no depende de un único parámetro, sino del balance entre adaptación, cobertura y facilidad de despliegue. En una red con múltiples receptores para monitoreo de gasoductos, la solución más robusta será aquella que conserve buen acoplamiento y cobertura útil aun con variaciones de instalación y orientación de nodos.

## 5. Evidencias de funcionamiento del enlace
La validación del enlace se completó con resultados numéricos de presupuesto de potencia mediante los modelos Friis y Okumura-Hata. El modelo de Friis proporciona una referencia ideal en espacio libre, útil como límite superior teórico. No obstante, para conclusiones de ingeniería aplicada debe priorizarse Okumura-Hata en entorno rural, al incorporar pérdidas de propagación más representativas de escenarios reales.

Con los parámetros de simulación empleados (f = 868 MHz, P_TX = 14 dBm, sensibilidad de -137 dBm y antenas seleccionadas), se obtiene margen de enlace positivo en la distancia de análisis. Además, los scripts calculan distancia máxima teórica y distancia máxima práctica al introducir margen de diseño adicional, lo cual aporta una medida de robustez frente a desvanecimientos y pérdidas no ideales.

Resultados numéricos del modelo Okumura-Hata (d = 10 km):
- $L_{p,urbano}(d)=161.22$ dB.
- $L_{p,rural}(d)=132.87$ dB.
- $P_R(d)=-114.67$ dBm.
- Margen de enlace = 22.33 dB.
- Distancia máxima teórica = 43.06 km.
- Distancia máxima práctica (margen de diseño 10 dB) = 11.65 km.

Interpretación: el enlace es viable con buen margen en 10 km por lado, y la distancia práctica estimada (11.65 km) es coherente con el escenario de gateway central para un trazado total de 20 km.

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

En esta sección se indica la ubicación de cada figura de simulación. Debajo de cada pie se inserta la imagen correspondiente y se mantiene la numeración establecida.

### 10.1 Figuras de script_monopolo

Figura 1. Geometría 3D de la antena monopolo λ/4 diseñada para 868 MHz.
Conclusión breve: La geometría confirma una solución compacta y apta para integración del gateway, preservando la cobertura omnidireccional necesaria para compatibilidad con sensores distribuidos.

Figura 2. Impedancia de entrada del monopolo λ/4 en la banda de análisis (Frecuencia [MHz] vs Impedancia [ohm]).
Conclusión breve: La cercanía de la impedancia a 50 ohm en torno a 868 MHz indica buen acoplamiento y compatibilidad eléctrica con el front-end RF del radioenlace.

Figura 3. Coeficiente de reflexión |S11| del monopolo λ/4 (Frecuencia [MHz] vs |S11| [dB]).
Conclusión breve: Un valor de |S11| por debajo de -10 dB alrededor de la frecuencia objetivo valida la adaptación en banda y reduce pérdidas por reflexión en el enlace.

Figura 4. Patrón de radiación 3D del monopolo a 868 MHz.
Conclusión breve: El diagrama evidencia comportamiento cuasi omnidireccional, condición clave para que el gateway mantenga compatibilidad espacial con sensores en distintos azimuts.

Figura 5. Patrón de radiación en azimut del monopolo a 868 MHz (Azimut [grados] vs Ganancia [dBi]).
Conclusión breve: La cobertura azimutal amplia y homogénea asegura compatibilidad del gateway con sensores ubicados en ambos sentidos del gasoducto.

Figura 6. Patrón de radiación en elevación del monopolo a 868 MHz (Elevación [grados] vs Ganancia [dBi]).
Conclusión breve: El patrón en elevación confirma ganancia útil para los ángulos de llegada de sensores cercanos y lejanos, sosteniendo la compatibilidad angular del enlace.

Figura 7. Distribución de corriente en el monopolo a 868 MHz.
Conclusión breve: La distribución confirma el modo de radiación esperado del monopolo y la coherencia física del modelo de gateway empleado en el radioenlace.

Figura 8. Comparación de ganancia entre variantes monopolo λ/4, monopolo λ/2 y dipolo λ/2.
Conclusión breve: La comparativa muestra el compromiso entre compacidad y ganancia, apoyando la selección del monopolo como antena de gateway para compatibilidad de cobertura.

Figura 9. Comparación de eficiencia estimada entre variantes monopolo λ/4, monopolo λ/2 y dipolo λ/2.
Conclusión breve: La eficiencia relativa permite identificar la alternativa más robusta cuando se prioriza rendimiento energético y estabilidad del enlace en despliegues prolongados.

### 10.2 Figuras de pachmicrostrip

Figura 10. Geometría 3D de la antena patch microstrip diseñada para 868 MHz.
Conclusión breve: La geometría plana facilita integración mecánica en sensores fijos y permite orientar cada patch hacia el gateway para mejorar compatibilidad direccional.

Figura 11. Impedancia de entrada del patch microstrip en banda (Frecuencia [MHz] vs Impedancia [ohm]).
Conclusión breve: El ajuste del punto de alimentación aproxima la impedancia a 50 ohm y asegura compatibilidad eléctrica del sensor con el módulo LoRa.

Figura 12. Coeficiente de reflexión |S11| del patch microstrip (Frecuencia [MHz] vs |S11| [dB]).
Conclusión breve: La profundidad del mínimo de |S11| alrededor de 868 MHz confirma sintonía y minimiza pérdidas de potencia en la transmisión sensor-gateway.

Figura 13. Patrón de radiación 3D del patch microstrip a 868 MHz.
Conclusión breve: El patrón más direccional respecto al monopolo es ventajoso en sensores fijos orientados al gateway, mejorando la compatibilidad del radioenlace en la dirección útil.

Figura 14. Patrón en azimut del patch microstrip a 868 MHz (Azimut [grados] vs Ganancia [dBi]).
Conclusión breve: La respuesta angular en azimut identifica la ventana de orientación óptima del patch para mantener compatibilidad con la posición del gateway.

Figura 15. Patrón en elevación del patch microstrip a 868 MHz (Elevación [grados] vs Ganancia [dBi]).
Conclusión breve: El corte en elevación permite verificar la orientación óptima del patch para mantener compatibilidad con los ángulos reales de llegada al gateway.

Figura 16. Distribución de corriente superficial en el patch microstrip a 868 MHz.
Conclusión breve: La corriente superficial valida el modo resonante esperado y respalda la coherencia electromagnética del sensor dentro del radioenlace completo.

### Conclusión del apartado 10
La secuencia de Figuras 1 a 16 proporciona evidencia completa de geometría, adaptación y radiación para ambas soluciones de antena. Esta trazabilidad visual permite justificar de forma sólida la elección final del sistema radiante para monitoreo distribuido en gasoductos.

