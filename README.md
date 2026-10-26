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

## Archivos de salida

El pronóstico se exporta como un archivo CSV elegido por el usuario. Incluye fecha, día de la semana, venta pronosticada, límites inferior y superior, y demanda acumulada. El archivo no incluye las métricas ni la recomendación de inventario; estas se muestran en la interfaz.

## Conclusión

El programa ofrece una forma práctica de explorar la demanda y apoyar decisiones de inventario mediante pronósticos y métricas visibles. Sus resultados son estimaciones basadas en el historial disponible, por lo que deben revisarse junto con el contexto operativo antes de realizar pedidos reales.
