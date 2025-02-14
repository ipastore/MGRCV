## NN

Que de las primeras observaciones vemos como la incertidumbre de la medida (verde) es radial. Estaria bien decirlo en el report. Y estaria bien saber por qué la incertidumbre de la odometria es mucho mayor.

Para analizar NN estaria bien tratar de cuantificar de alguna manera con alguna grafica el numero de false positive y negatives a lo largo de todo el codigo. HECHO!

Habria que hacer una comparación y un analisis de NN con y sin personas. Y quizas tratar de definir en que tipo de situaciones se hace un lio NN cuando tiene dos candidatos.

## SINGLE

Guardamos en H para cada observación el indice de la feature a la que corresponde. Si no tiene correspondencia se le asigna 0.

El SINGLE parece ser aparentemente más rbusuto que el NN sin personas. No obstante, no tenemos en cuenta ningun criterio de distancia ni nada parecido. ¿Te elimina por completo los false positive?

COn personas vemos que reacciona peor. Habrá que describir el por qué. Seguramente tenga que ver por el filtro de distancia.

Estaria bien plantear una fusión de SINGLE con NN para ver su funcionmiento.