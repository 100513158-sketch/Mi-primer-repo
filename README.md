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

La lógica de decisión adoptada en esta memoria se estructura en dos barridos complementarios y luego una validación final. Primero se ejecuta un barrido conservador de canal con Okumura-Hata y ganancias fijas (script barrido_okhata_ganancias_fijas.m) para fijar una cota base de altura mínima. Como validación intermedia, se ejecuta un segundo barrido con Okumura-Hata y ganancias reales dependientes del ángulo de elevación (script barrido_okhata_antenas_reales.m) para cuantificar el efecto de la respuesta angular de las antenas reales. Finalmente, se ejecuta el barrido electromagnético con antenas diseñadas, dependencia angular completa y optimización de tilt (script barrido_gateway_gasoducto.m) para validar el enlace con patrones reales, orientación y peor caso azimutal. Con esta metodología de tres etapas, el barrido conservador arroja h_min,10 = 8 m y h_min,15 = 14 m en 10 km; el barrido con ganancias reales proporciona validación intermedia con dependencia angular; y el barrido electromagnético confirma viabilidad robusta y mejora de margen por ganancia efectiva y orientación. A efectos de ingeniería, se distingue explícitamente entre altura mínima funcional (10 dB) y altura mínima robusta (15 dB).

El análisis geométrico confirma que los ángulos de llegada en elevación son bajos en todo el trazado (aprox. 0.49 deg a 1 km y 0.05 deg a 10 km para $h_{GW}=10$ m y $h_S=1.5$ m). En consecuencia, la cobertura efectiva queda gobernada por la respuesta del gateway cerca del horizonte: cualquier depresión de ganancia en elevaciones bajas penaliza primero el extremo lejano. Por ello, además de validar azimut, se debe verificar explícitamente el patrón de elevación en ese intervalo angular y ajustar altura/tilt del gateway si se detectan degradaciones en el peor caso.

### 2.1 Análisis geométrico del ángulo de llegada (sensor cercano frente a lejano)
Para cuantificar el efecto anterior se emplea una aproximación geométrica básica del ángulo de elevación de llegada al gateway:

$$
θ = \arctan\left(\frac{h_{GW} - h_{S}}{d}\right)
$$

donde $h_{GW}$ es la altura de la antena del gateway, $h_S$ la altura de la antena del sensor y $d$ la distancia horizontal entre ambos.

Para el análisis geométrico se adopta $h_{GW}=10$ m como caso de diseño de referencia para representar el escenario optimizado con antenas diseñadas, y $h_S=1.5$ m. Con ello, la diferencia de altura es $\Delta h = 8.5$ m:

- Sensor cercano a $d=1$ km:

$$
θ_{1km}=\arctan\left(\frac{8.5}{1000}\right)\approx 0.49^\circ
$$

- Sensor lejano a $d=10$ km:

$$
θ_{10km}=\arctan\left(\frac{8.5}{10000}\right)\approx 0.05^\circ
$$

Por tanto, la parte más crítica del patrón del gateway es la zona de elevaciones bajas. Si la antena presenta depresión de ganancia cerca del horizonte, el sensor más lejano será el primero en degradarse. Este análisis justifica verificar con detalle el patrón de elevación del gateway y, si fuese necesario, aumentar altura de instalación o ajustar inclinación mecánica para mejorar cobertura en el extremo del trazado.

Como referencia conservadora de robustez (umbral 15 dB) obtenida en el barrido con ganancias fijas, también puede considerarse $h_{GW}=14$ m. En ese caso, con $h_S=1.5$ m se tiene $\Delta h=12.5$ m, y los ángulos característicos son aproximadamente $0.72^\circ$ (1 km) y $0.07^\circ$ (10 km).

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
- Se valida ganancia útil en el rango de ángulos de llegada calculado (aprox. $0.05^\circ$ a $0.49^\circ$ para 10 km y 1 km, respectivamente, con $h_{GW}=10$ m y $h_S=1.5$ m).
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
En esta memoria se separan tres niveles de análisis:
- Barrido A (conservador de propagación): Okumura-Hata con ganancias fijas (script barrido_okhata_ganancias_fijas.m).
- Barrido A+ (conservador con antenas reales): Okumura-Hata con ganancias reales dependientes de ángulo de elevación (script barrido_okhata_antenas_reales.m).
- Barrido B (electromagnético del sistema): Okumura-Hata + patrones reales de antena + tilt + peor caso azimutal (script barrido_gateway_gasoducto.m).

Esta separación en tres niveles evita mezclar hipótesis y permite justificar de forma trazable la altura mínima funcional y robusta, proporcionando validación incremental desde el caso más conservador hasta el más optimizado.

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

Resultados del Barrido A (ganancias fijas, modelo conservador):
- Umbral funcional (10 dB): altura mínima $h_{min,10}=8$ m.
- Umbral robusto (15 dB): altura mínima $h_{min,15}=14$ m.
- En $h_{TX}=10$ m: margen en 10 km = 12.61 dB (cumple 10 dB, no cumple 15 dB).

Resultados del Barrido A+ (ganancias reales de antena, modelo conservador con dependencia angular):
- Umbral funcional (10 dB): altura mínima $h_{min,10}=6$ m (margen = 10.58 dB).
- Umbral robusto (15 dB): altura mínima $h_{min,15}=10$ m (margen = 15.25 dB).
- En $h_{TX}=10$ m: margen en 10 km = 15.25 dB (cumple 10 dB y cumple 15 dB).
- Este barrido valida que el uso de ganancias reales dependientes del ángulo de elevación reduce significativamente la altura requerida (2 m menos para funcional, 4 m menos para robusto) respecto al modelo de ganancias fijas.

Resultados del Barrido B (antenas diseñadas, caso $L_{pol}=0$ dB):
- Ganancia efectiva del sensor patch (broadside): 9.76 dBi.
- Viabilidad robusta del enlace confirmada en el extremo de 10 km para configuraciones del barrido.
- Altura mínima robusta (criterio 15 dB a 10 km): $h_{GW}=9$ m, tilt = -1 deg, margen en 10 km = 15.39 dB.
- Caso representativo robusto: $h_{GW}=10$ m, tilt = -1 deg, margen en 10 km = 16.33 dB.
- Mejor configuración por peor caso: $h_{GW}=15$ m, tilt = -1 deg, margen peor caso = 19.91 dB (margen en 10 km = 19.91 dB).

Definición explícita de altura mínima en el barrido:
- Para cada altura barrida, se selecciona el tilt que maximiza el margen en 10 km.
- Con esa curva "mejor margen en 10 km vs altura", se toma la primera altura que cumple el umbral objetivo.
- En símbolos: $h_{min}=\min\{h: \max_{tilt} M(d=10\text{ km},h,tilt) \ge M_{obj}\}$.
- En el Barrido A (ganancias fijas), se obtiene $h_{min,10}=8$ m y $h_{min,15}=14$ m.
- En el Barrido B (antenas diseñadas), $h_{min}$ depende del patrón, tilt y criterio de peor caso azimutal.

Interpretación y criterio final de diseño:
- Si se impone criterio robusto conservador (15 dB) con hipótesis de ganancias fijas, la altura recomendada es 14 m.
- Si se adopta la validación electromagnética con antenas diseñadas y orientación controlada, 10 m resulta una solución técnicamente viable con margen holgado.
- En todos los casos, la memoria debe mantener explícita la diferencia entre "mínimo funcional" (10 dB) y "mínimo robusto" (15 dB).

Figura 21. Okumura-Hata (ganancias fijas): margen en 10 km en función de la altura del gateway.
Insertar aquí la figura correspondiente: curva de margen en 10 km vs altura, con marcado de h_min,10 y h_min,15.
Conclusión breve: La figura ilustra cómo el margen de enlace en el extremo lejano (10 km) aumenta linealmente con la altura del gateway en el modelo Okumura-Hata con ganancias fijas (2 dBi / 2.2 dBi). Se identifica claramente h_min,10 = 8 m como la altura mínima funcional y h_min,15 = 14 m como la altura mínima robusta (15 dB).

Figura 22. Okumura-Hata (ganancias fijas): perfiles de margen vs distancia para alturas representativas.
Insertar aquí la figura correspondiente: curvas de margen vs distancia para h = 10 m, 14 m y 20 m, con líneas de umbral.
Conclusión breve: La comparación de perfiles muestra cómo el margen en el extremo (10 km) mejora con altura, confirmando que h = 14 m es suficiente para alcanzar 15 dB de margen robusto en el modelo conservador de canal. En cambio, h = 10 m alcanza solo 12.61 dB en el extremo, situándose entre los dos umbrales de decisión.

Figura 23. Okumura-Hata (ganancias REALES de antena): margen en 10 km en función de la altura del gateway.
Insertar aquí la figura correspondiente: curva de margen en 10 km vs altura con ganancias reales del monopolo y patch, con marcado de h_min,10 y h_min,15.
Conclusión breve: Al usar las ganancias reales del monopolo y patch microstrip (que varían con el ángulo de elevación de llegada), se observa que el margen de enlace evoluciona de forma significativamente diferente al caso de ganancias fijas. Los resultados muestran h_min,10 = 6 m y h_min,15 = 10 m, mejorando en 2 m y 4 m respectivamente respecto a Barrido A. Esta validación intermedia demuestra que la dependencia angular real aporta mejora sustancial sin necesidad de optimización adicional como tilt.

Figura 24. Okumura-Hata (ganancias REALES de antena): perfiles de margen vs distancia para alturas representativas.
Insertar aquí la figura correspondiente: curvas de margen vs distancia para h = 5 m, 10 m, 14 m y 20 m con ganancias reales de antena.
Conclusión breve: La comparación de perfiles con ganancias reales demuestra cómo la respuesta angular de las antenas afecta a la distribución de margen a lo largo del trazado. El perfil a h = 10 m muestra margen de 15.25 dB en el extremo (10 km), cumpliendo tanto el umbral funcional como el robusto. La progresión suave de márgenes refleja la ganancia variable con ángulo, capturando el efecto realista de la respuesta de las antenas a elevaciones bajas.

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
- Caso $L_{pol}=0$ dB (configuración robusta de referencia): margen peor caso = 19.91 dB; margen a 10 km = 19.91 dB.
- Caso $L_{pol}=3$ dB (inferencia del modelo): margen peor caso = 16.91 dB; margen a 10 km = 16.91 dB.
- Configuración robusta en ambos casos: se mantiene la misma geometría del barrido de referencia (misma altura y mismo tilt), variando únicamente $L_{pol}$.

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
La validación del enlace se completó con resultados numéricos de presupuesto de potencia mediante los modelos Friis y Okumura-Hata, ambos evaluados con la misma configuración de diseño para permitir comparación directa. El modelo de Friis proporciona una referencia ideal en espacio libre (límite superior teórico), mientras que Okumura-Hata rural se toma como base de decisión de ingeniería por representar mejor la propagación del escenario.

Con los parámetros de simulación empleados (f = 868 MHz, P_TX = 14 dBm, sensibilidad de -137 dBm y antenas seleccionadas), se obtiene margen de enlace positivo en la distancia de análisis. Además, los scripts calculan distancia máxima teórica y distancia máxima práctica al introducir margen de diseño adicional, lo cual aporta una medida de robustez frente a desvanecimientos y pérdidas no ideales.

Resultados numéricos del modelo Friis (d = 10 km):
- $L_p(d)=111.21$ dB.
- $P_R(d)=-93.01$ dBm.
- Margen de enlace = 43.99 dB.
- Distancia máxima teórica = 1583.29 km.
- Distancia máxima práctica (margen de diseño 10 dB) = 561.61 km.

Resultados numéricos del modelo Okumura-Hata (d = 10 km):
- $L_{p,urbano}(d)=170.94$ dB.
- $L_{p,rural}(d)=142.59$ dB.
- $P_R(d)=-124.39$ dBm.
- Margen de enlace = 12.61 dB.
- Distancia máxima teórica = 21.33 km.
- Distancia máxima práctica (margen de diseño 10 dB) = 11.70 km.

Comparación Friis vs Okumura-Hata en 10 km:
- La potencia recibida en Okumura-Hata es 31.38 dB menor que en Friis ($-124.39$ dBm frente a $-93.01$ dBm).
- El margen también disminuye en la misma cuantía (12.61 dB frente a 43.99 dB).
- Por ello, Friis se usa solo como cota ideal y Okumura-Hata como criterio principal para aceptación del enlace.

Interpretación: con $h_{TX}=10$ m el enlace es viable para criterio funcional de 10 dB, pero no alcanza el criterio robusto de 15 dB en el modelo conservador de ganancias fijas (con $G_{TX}=2$ dBi y $G_{RX}=2.2$ dBi). El barrido de sensibilidad de altura en este modelo sitúa el umbral de 15 dB en $h_{TX}=14$ m, y la distancia práctica estimada (11.70 km) sigue siendo coherente con el escenario de gateway central para un trazado total de 20 km.

La evidencia completa de correcto funcionamiento debe incluir, en el documento final, tres bloques: diagramas de directividad, representación de |S11| y resultados numéricos del balance de enlace (potencia recibida, margen y alcance estimado).

### Conclusiones del apartado 5
El enlace puede considerarse funcional cuando la potencia recibida supera la sensibilidad con margen positivo y las antenas se mantienen adaptadas en banda. Para diseño robusto, el criterio de 15 dB debe verificarse con el modelo declarado.

La validación del enlace se estructura en tres niveles complementarios:

1. **Barrido A (conservador de propagación)**: Okumura-Hata con ganancias fijas (2 dBi / 2.2 dBi). Proporciona una cota conservadora e independiente de patrones de antena específicos. Resultados: $h_{min,10}=8$ m, $h_{min,15}=14$ m (Figuras 21–22).

2. **Barrido A+ (conservador con dependencia angular)**: Okumura-Hata con ganancias reales del monopolo y patch microstrip, que varían según ángulo de elevación. Permite cuantificar el impacto de la respuesta angular sin optimización de tilt. Resultados: $h_{min,10}=6$ m (margen 10.58 dB), $h_{min,15}=10$ m (margen 15.25 dB) (Figuras 23–24). La mejora respecto a Barrido A (2 m menos en umbral funcional y 4 m menos en umbral robusto) demuestra el impacto significativo de usar ganancias reales dependientes del ángulo.

3. **Barrido B (electromagnético completo)**: Okumura-Hata + patrones reales de antena + tilt + peor caso azimutal (script barrido_gateway_gasoducto.m). Proporciona validación electromagnética completa con todas las variables del sistema. Resultados: altura mínima robusta de 9 m con tilt -1 deg (15.39 dB a 10 km), caso representativo en h=10 m con tilt -1 deg (16.33 dB) y mejor configuración por peor caso en h=15 m con tilt -1 deg (19.91 dB).

El análisis del Barrido A proporciona una cota conservadora de altura requerida: $h_{min,10}=8$ m para funcionalidad y $h_{min,15}=14$ m para robustez. Este barrido aísla la contribución del modelo de propagación, eliminando dependencias de patrones específicos de antena y permitiendo una evaluación pura del canal en el escenario rural del gasoducto. 

El Barrido A+ valida que al introducir ganancias reales del monopolo y patch que varían con el ángulo de elevación (sin optimización adicional como tilt), la altura requerida mejora significativamente: $h_{min,10}=6$ m (2 m de reducción) y $h_{min,15}=10$ m (4 m de reducción). Esta mejora demuestra que el modelo de ganancias fijas es sobradamente conservador, y que la dependencia angular real del patrón de las antenas contribuye notablemente a mejorar el margen de enlace disponible.

El Barrido B (barrido_gateway_gasoducto.m con patrones reales, tilt optimizado y análisis de peor caso azimutal) demuestra que con antenas diseñadas, optimización de orientación y validación de cobertura angular, se alcanza robustez (≥15 dB) desde h=9 m con tilt -1 deg (15.39 dB a 10 km), y con h=10 m se obtiene margen robusto de 16.33 dB. Además, la configuración que maximiza el peor caso en el barrido es h=15 m con tilt -1 deg (19.91 dB).

Por tanto, la validación completa del enlace exige tres etapas que proporcionan mejora progresiva: primero verificación conservadora de propagación pura con ganancias fijas (Barrido A, h_min,15 = 14 m), segundo validación de dependencia angular con antenas reales (Barrido A+, h_min,15 = 10 m), y tercero verificación electromagnética completa del sistema con tilt optimizado (Barrido B, h_min,15 = 9 m). Los tres niveles son necesarios para justificar técnicamente la altura de instalación del gateway según el nivel de conservadurismo requerido y para demostrar la mejora aportada por cada nivel de validación.

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

Como cierre de diseño, la memoria adopta explícitamente dos niveles de decisión: criterio funcional de 10 dB y criterio robusto de 15 dB, validados mediante tres barridos complementarios. Bajo enfoque conservador de propagación (ganancias fijas), el mínimo robusto exige 14 m; bajo validación con ganancias reales dependientes del ángulo, se obtiene validación intermedia; bajo validación electromagnética con antenas diseñadas y optimización de orientación, la solución de 10 m presenta margen holgado en el caso modelado. Esta triple validación permite defender la decisión final de altura según el nivel de conservadurismo requerido por operación y despliegue.

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



