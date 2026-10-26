# Sistema de pronóstico de demanda e inventario

Aplicación de escritorio en Python para analizar ventas diarias por tienda y artículo, proyectar la demanda de los próximos días y estimar una cantidad de reabastecimiento. La interfaz está construida con Tkinter y presenta resultados en gráficos, indicadores y una tabla exportable.

## Funcionalidades

- Genera datos sintéticos de demostración al iniciar, sin necesidad de un archivo externo.
- Permite cargar un archivo CSV de ventas con las columnas `date`, `store`, `item` y `sales`. No importa si los nombres vienen en mayúsculas; el programa los normaliza.
- Filtra los datos por tienda y artículo.
- Ofrece cuatro modelos: regresión lineal múltiple, Ridge, Lasso y Random Forest.
- Pronostica entre 7 y 120 días. Para ello usa fechas, tendencias, ventas anteriores y promedios móviles.
- Evalúa el modelo con MAE, RMSE, MAPE y R², reservando los últimos 90 días disponibles para la prueba.
- Estima la demanda total, el stock de seguridad y el pedido sugerido según el inventario actual y un nivel de confianza de 90 %, 95 % o 99 %.
- Muestra el pronóstico y los resultados en gráficos y una tabla; permite exportar esta última a CSV.

## Requisitos

- Python 3.10 o posterior.
- Tkinter, necesario para la interfaz gráfica. En muchas instalaciones de Python ya viene incluido.
- Bibliotecas de Python: `pandas`, `numpy`, `scikit-learn` y `matplotlib`.
- Un entorno de escritorio con soporte gráfico para abrir la ventana de la aplicación.

## Instalación en Linux

Abre una terminal en la carpeta del proyecto, donde está `sistema_forecasting.py`, y ejecuta:

1. Comprueba que Python y `venv` estén instalados:

   ```bash
   python3 --version
   ```

   En Debian o Ubuntu, si hace falta instalar Python, `venv` o Tkinter:

   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-tk
   ```

2. Crea un entorno virtual:

   ```bash
   python3 -m venv venv
   ```

3. Activa el entorno virtual:

   ```bash
   source venv/bin/activate
   ```

4. Actualiza `pip` e instala las bibliotecas requeridas:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install pandas numpy scikit-learn matplotlib
   ```

## Instalación en Windows

Desde PowerShell, en la carpeta del proyecto:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn matplotlib
```

Tkinter normalmente se incluye con la instalación oficial de Python para Windows. Si PowerShell bloquea la activación del entorno, se puede abrir `cmd` y activar con `venv\Scripts\activate.bat`.

## Ejecución

Con el entorno virtual activado y desde la carpeta del proyecto, ejecuta:

```bash
python sistema_forecasting.py
```

En Windows se usa el mismo comando después de activar el entorno. Se abrirá la ventana del sistema con datos sintéticos ya cargados y un primer pronóstico calculado.

## Uso paso a paso

1. Inicia el programa. Para empezar a probarlo, conserva los datos demo o pulsa **Generar Dataset Demo**.
2. Para trabajar con tus datos, pulsa **Cargar train.csv (Kaggle)** y selecciona el CSV. Debe contener una fila por fecha, tienda y artículo, con las columnas `date`, `store`, `item` y `sales`. `date` debe ser interpretable como fecha y `sales` debe contener las ventas diarias.
3. Selecciona la tienda y el artículo que quieres analizar.
4. Elige el algoritmo de regresión y el horizonte de pronóstico, entre 7 y 120 días.
5. Introduce el inventario físico actual y selecciona el nivel de servicio deseado.
6. Pulsa **Ejecutar Forecasting** para actualizar los indicadores, los gráficos y la tabla.
7. Consulta la gráfica de pronóstico, la importancia o los coeficientes de las variables y el detalle diario. En la pestaña de la tabla, pulsa **Exportar Pronóstico a CSV** para guardar los resultados.

El sistema requiere al menos 120 días históricos para procesar una serie. Como utiliza los últimos 90 días para evaluar el modelo, conviene proporcionar un historial considerablemente más largo para que el entrenamiento tenga suficientes datos.

## Descargar e integrar el dataset

El programa es compatible con el archivo `train.csv` del reto **Store Item Demand Forecasting Challenge** de Kaggle. Se puede descargar desde la [página oficial de datos del reto](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data). Kaggle puede solicitar iniciar sesión y aceptar las reglas de la competencia antes de habilitar la descarga.

1. Abre el enlace anterior y descarga `train.csv` (o usa **Download All** y extrae el archivo ZIP).
2. Ejecuta `sistema_forecasting.py`.
3. Pulsa **Cargar train.csv (Kaggle)** y selecciona el archivo descargado.
4. El programa valida y normaliza los encabezados, carga las ventas y ejecuta el pronóstico. Después puedes cambiar tienda, artículo y parámetros desde la interfaz.

El CSV debe incluir estas columnas:

| Columna | Contenido esperado |
| --- | --- |
| `date` | Fecha de las ventas, interpretable como fecha |
| `store` | Identificador de la tienda |
| `item` | Identificador del artículo |
| `sales` | Unidades vendidas ese día |

Se necesita un mínimo de 120 registros para la tienda y artículo seleccionados. El modelo reserva los últimos 90 días para evaluación, así que un historial más largo permite entrenarlo con más datos. El archivo de Kaggle contiene el historial agregado por fecha, tienda y artículo; el sistema carga el CSV completo y permite filtrar una combinación en la aplicación.

## Guía de la interfaz

### Controles laterales

- **Origen del dataset:** carga el archivo CSV de Kaggle o reemplaza los datos actuales con datos sintéticos de demostración.
- **Tienda y artículo:** limitan el análisis a una serie de ventas concreta.
- **Algoritmo y horizonte:** seleccionan el modelo de regresión y cuántos días futuros se calcularán (de 7 a 120).
- **Inventario físico y nivel de servicio:** indican las unidades disponibles y el nivel de confianza usado para estimar el stock de seguridad.
- **Ejecutar Forecasting:** vuelve a entrenar y actualiza las métricas, gráficas y tabla con los parámetros seleccionados.

### Indicadores superiores

- **Error MAE:** error absoluto promedio del modelo en los días de prueba; se expresa en unidades. El porcentaje secundario muestra el MAPE.
- **Error RMSE / R²:** RMSE resume el error y penaliza más las desviaciones grandes; R² indica cuánto se ajusta el modelo a los datos de prueba.
- **Demanda pronosticada:** suma de las ventas estimadas para todo el horizonte elegido.
- **Stock de seguridad:** inventario adicional calculado a partir del RMSE, el horizonte y el nivel de servicio.
- **Pedido sugerido:** compara la demanda pronosticada más el stock de seguridad con el inventario físico. Si el inventario alcanza, muestra el superávit en lugar de un pedido.

### Pestañas de resultados

- **Gráfica de Pronóstico:** compara las ventas reales de prueba con el ajuste del modelo y muestra la proyección futura. La franja sombreada representa el intervalo estimado alrededor del pronóstico.
- **Pesos de Variables:** ayuda a interpretar qué variables influyen en el modelo. En regresiones muestra coeficientes positivos o negativos; en Random Forest muestra importancia relativa.
- **Tabla de Pronóstico Diario y Exportación:** presenta la fecha, el día, la venta estimada, los límites inferior y superior y la demanda acumulada. El botón **Exportar Pronóstico a CSV** guarda estas columnas en un archivo.

## Capturas del sistema

### Pronóstico de demanda

![Captura de la gráfica de pronóstico](<assets/Captura desde 2026-10-25 21-28-48.png>)

La gráfica compara las ventas reales de los últimos 90 días con el ajuste del modelo y proyecta las ventas futuras para la tienda y el artículo seleccionados. La banda naranja representa el intervalo estimado; en la parte superior se resumen los errores, la demanda total, el stock de seguridad y el pedido sugerido.

### Pesos de variables

![Captura de los coeficientes del modelo](<assets/Captura desde 2026-10-25 21-29-06.png>)

Esta vista muestra los coeficientes de la regresión lineal para las variables usadas por el modelo. Las barras azules representan una relación positiva y las rojas una relación negativa; las variables con barras de mayor magnitud tienen mayor influencia en esta regresión.

### Tabla de pronóstico diario

![Captura de la tabla de pronóstico y exportación](<assets/Captura desde 2026-10-25 21-29-12.png>)

La tabla detalla, día por día, las unidades pronosticadas, los límites mínimo y máximo y la demanda acumulada durante el horizonte seleccionado. El botón **Exportar Pronóstico a CSV** permite guardar estos resultados.

## Archivos de salida

El pronóstico se exporta como un archivo CSV elegido por el usuario. Incluye fecha, día de la semana, venta pronosticada, límites inferior y superior, y demanda acumulada. El archivo no incluye las métricas ni la recomendación de inventario; estas se muestran en la interfaz.

## Conclusión

El sistema facilita la consulta de pronósticos y necesidades de inventario a partir de ventas históricas. Sus resultados son estimaciones y conviene contrastarlos con el contexto operativo antes de realizar pedidos reales.
