import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import timedelta

# Modelos de Regresión y Métricas de IA (Scikit-Learn)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Integración de Matplotlib en Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class SistemaForecastingIA:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Predictivo de Inventario y Ventas - IA (Forecasting)")
        self.root.geometry("1360x820")
        self.root.minsize(1150, 720)

        # Variables de estado y datos
        self.df_raw = None
        self.df_forecast_result = None
        self.feature_cols = [
            'time_idx', 'dayofweek', 'month', 'is_weekend',
            'sin_dayofyear', 'cos_dayofyear',
            'lag_1', 'lag_7', 'lag_14', 'lag_28',
            'rolling_mean_7', 'rolling_mean_28', 'rolling_std_7'
        ]

        self._configurar_estilos()
        self._construir_interfaz()

        # Generar datos demo al iniciar para que el sistema sea usable de inmediato
        self.generar_datos_demo(mostrar_alerta=False)

    def _configurar_estilos(self):
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        style.configure("TFrame", background="#f4f6f9")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#f4f6f9", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#1e293b", foreground="#ffffff", font=("Segoe UI", 14, "bold"))
        style.configure("SubHeader.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 11, "bold"))
        style.configure("MetricTitle.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 9, "bold"))
        style.configure("MetricValue.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 15, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _construir_interfaz(self):
        # Barra superior
        header_frame = tk.Frame(self.root, bg="#1e293b", height=55)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        lbl_titulo = ttk.Label(
            header_frame,
            text="  SISTEMA INTELIGENTE DE PRONÓSTICO DE DEMANDA Y GESTIÓN DE INVENTARIO",
            style="Header.TLabel"
        )
        lbl_titulo.pack(side=tk.LEFT, padx=15, pady=10)

        self.lbl_estado_datos = tk.Label(
            header_frame, text="Datos: Demo cargado", bg="#1e293b", fg="#38bdf8", font=("Segoe UI", 10, "bold")
        )
        self.lbl_estado_datos.pack(side=tk.RIGHT, padx=20)

        # Contenedor principal
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        # ---------------- PANEL IZQUIERDO (CONTROLES) ----------------
        sidebar = ttk.Frame(main_container, style="Card.TFrame", padding=15, width=320)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        # 1. Origen de Datos
        ttk.Label(sidebar, text="1. Origen del Dataset", style="SubHeader.TLabel").pack(anchor=tk.W, pady=(0, 8))
        btn_cargar = ttk.Button(sidebar, text="📂 Cargar train.csv (Kaggle)", command=self.cargar_csv_kaggle)
        btn_cargar.pack(fill=tk.X, pady=3)

        btn_demo = ttk.Button(sidebar, text="⚡ Generar Dataset Demo", command=lambda: self.generar_datos_demo(True))
        btn_demo.pack(fill=tk.X, pady=3)

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        # 2. Selección de Tienda y Artículo
        ttk.Label(sidebar, text="2. Filtro de Producto", style="SubHeader.TLabel").pack(anchor=tk.W, pady=(0, 8))

        tk.Label(sidebar, text="Tienda (Store ID):", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.combo_store = ttk.Combobox(sidebar, state="readonly")
        self.combo_store.pack(fill=tk.X, pady=(2, 8))

        tk.Label(sidebar, text="Artículo (Item ID):", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.combo_item = ttk.Combobox(sidebar, state="readonly")
        self.combo_item.pack(fill=tk.X, pady=(2, 8))

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        # 3. Configuración del Modelo de IA
        ttk.Label(sidebar, text="3. Modelo de Forecasting (IA)", style="SubHeader.TLabel").pack(anchor=tk.W, pady=(0, 8))

        tk.Label(sidebar, text="Algoritmo de Regresión:", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.combo_modelo = ttk.Combobox(
            sidebar,
            state="readonly",
            values=[
                "Regresión Lineal Múltiple (OLS)",
                "Regresión Ridge (Regularización L2)",
                "Regresión Lasso (Regularización L1)",
                "Random Forest Regressor"
            ]
        )
        self.combo_modelo.current(0)
        self.combo_modelo.pack(fill=tk.X, pady=(2, 8))

        tk.Label(sidebar, text="Horizonte a Pronosticar (días futuros):", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.spin_horizonte = ttk.Spinbox(sidebar, from_=7, to=120, increment=1)
        self.spin_horizonte.set(30)
        self.spin_horizonte.pack(fill=tk.X, pady=(2, 8))

        ttk.Separator(sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        # 4. Parámetros de Inventario
        ttk.Label(sidebar, text="4. Parámetros de Almacén", style="SubHeader.TLabel").pack(anchor=tk.W, pady=(0, 8))

        tk.Label(sidebar, text="Inventario Físico Actual (Unidades):", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.entry_stock_actual = ttk.Entry(sidebar)
        self.entry_stock_actual.insert(0, "1200")
        self.entry_stock_actual.pack(fill=tk.X, pady=(2, 8))

        tk.Label(sidebar, text="Nivel de Servicio (Stock Seguridad):", bg="#ffffff", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.combo_confianza = ttk.Combobox(
            sidebar,
            state="readonly",
            values=["90% (Z = 1.28)", "95% (Z = 1.65)", "99% (Z = 2.33)"]
        )
        self.combo_confianza.current(1)
        self.combo_confianza.pack(fill=tk.X, pady=(2, 14))

        # Botón Principal de Ejecución
        btn_ejecutar = tk.Button(
            sidebar,
            text="▶ EJECUTAR FORECASTING",
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            pady=10,
            cursor="hand2",
            command=self.ejecutar_pipeline_forecasting
        )
        btn_ejecutar.pack(fill=tk.X, pady=(5, 5))

        # ---------------- PANEL DERECHO (RESULTADOS Y GRÁFICAS) ----------------
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Fila de Tarjetas KPI (Métricas de IA + Decisión de Inventario)
        kpi_frame = ttk.Frame(right_panel)
        kpi_frame.pack(fill=tk.X, pady=(0, 10))

        self.kpi_mae = self._crear_tarjeta_kpi(kpi_frame, "ERROR MAE (Test)", "--", "Unidades promedio")
        self.kpi_rmse = self._crear_tarjeta_kpi(kpi_frame, "ERROR RMSE / R²", "--", "Precisión del modelo")
        self.kpi_demanda = self._crear_tarjeta_kpi(kpi_frame, "DEMANDA PRONOSTICADA", "--", "Próximos N días")
        self.kpi_safety = self._crear_tarjeta_kpi(kpi_frame, "STOCK DE SEGURIDAD", "--", "Colchón anti-quiebre")
        self.kpi_reorden = self._crear_tarjeta_kpi(kpi_frame, "PEDIDO SUGERIDO", "--", "Estado de almacén")

        # Notebook (Pestañas de visualización)
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Pestaña 1: Gráfica de Serie de Tiempo y Forecasting
        self.tab_grafica = ttk.Frame(self.notebook, style="Card.TFrame")
        self.notebook.add(self.tab_grafica, text="📈 Gráfica de Pronóstico (Forecasting)")

        self.fig_forecast = Figure(figsize=(9, 5), dpi=100, facecolor="#ffffff")
        self.ax_forecast = self.fig_forecast.add_subplot(111)
        self.canvas_forecast = FigureCanvasTkAgg(self.fig_forecast, master=self.tab_grafica)
        self.canvas_forecast.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        toolbar_frame = ttk.Frame(self.tab_grafica)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas_forecast, toolbar_frame)

        # Pestaña 2: Importancia de Variables (Explicabilidad de la Regresión)
        self.tab_pesos = ttk.Frame(self.notebook, style="Card.TFrame")
        self.notebook.add(self.tab_pesos, text="🧠 Pesos de Variables (Feature Importance)")

        self.fig_pesos = Figure(figsize=(9, 5), dpi=100, facecolor="#ffffff")
        self.ax_pesos = self.fig_pesos.add_subplot(111)
        self.canvas_pesos = FigureCanvasTkAgg(self.fig_pesos, master=self.tab_pesos)
        self.canvas_pesos.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Pestaña 3: Tabla de Datos Pronosticados y Exportación
        self.tab_tabla = ttk.Frame(self.notebook, style="Card.TFrame", padding=10)
        self.notebook.add(self.tab_tabla, text="📋 Tabla de Pronóstico Diario y Exportación")

        top_table_bar = ttk.Frame(self.tab_tabla, style="Card.TFrame")
        top_table_bar.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            top_table_bar,
            text="Detalle diario de unidades proyectadas e intervalos de confianza:",
            style="SubHeader.TLabel"
        ).pack(side=tk.LEFT)

        btn_exportar = ttk.Button(
            top_table_bar, text="💾 Exportar Pronóstico a CSV", command=self.exportar_pronostico_csv
        )
        btn_exportar.pack(side=tk.RIGHT)

        columnas = ("fecha", "dia_semana", "venta_estimada", "limite_inferior", "limite_superior", "demanda_acumulada")
        self.tree = ttk.Treeview(self.tab_tabla, columns=columnas, show="headings", height=18)

        encabezados = {
            "fecha": "Fecha Futura",
            "dia_semana": "Día de la Semana",
            "venta_estimada": "Venta Pronosticada (Unid.)",
            "limite_inferior": "Mínimo Esperado",
            "limite_superior": "Máximo Esperado",
            "demanda_acumulada": "Demanda Acumulada"
        }
        for col, texto in encabezados.items():
            self.tree.heading(col, text=texto)
            self.tree.column(col, anchor=tk.CENTER, width=140)

        scrollbar = ttk.Scrollbar(self.tab_tabla, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _crear_tarjeta_kpi(self, parent, titulo, valor_inicial, subtitulo):
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        lbl_title = ttk.Label(card, text=titulo, style="MetricTitle.TLabel")
        lbl_title.pack(anchor=tk.W)

        lbl_val = ttk.Label(card, text=valor_inicial, style="MetricValue.TLabel")
        lbl_val.pack(anchor=tk.W, pady=(4, 2))

        lbl_sub = tk.Label(card, text=subtitulo, bg="#ffffff", fg="#64748b", font=("Segoe UI", 8))
        lbl_sub.pack(anchor=tk.W)

        return {"val": lbl_val, "sub": lbl_sub}

    # =========================================================================
    # CARGA DE DATOS (KAGGLE CSV O GENERADOR SINTÉTICO)
    # =========================================================================
    def cargar_csv_kaggle(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo train.csv de Kaggle",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if not ruta:
            return

        try:
            df = pd.read_csv(ruta)
            columnas_requeridas = {'date', 'store', 'item', 'sales'}
            # Adaptación automática por si las columnas vienen en mayúsculas
            df.columns = [c.lower().strip() for c in df.columns]

            if not columnas_requeridas.issubset(set(df.columns)):
                messagebox.showerror(
                    "Formato Incompatible",
                    f"El CSV debe contener las columnas: {columnas_requeridas}\n"
                    f"Columnas encontradas: {list(df.columns)}"
                )
                return

            df['date'] = pd.to_datetime(df['date'])
            self.df_raw = df.sort_values('date').reset_index(drop=True)

            tiendas = sorted(self.df_raw['store'].unique().tolist())
            articulos = sorted(self.df_raw['item'].unique().tolist())

            self.combo_store['values'] = tiendas
            self.combo_item['values'] = articulos
            self.combo_store.current(0)
            self.combo_item.current(0)

            self.lbl_estado_datos.config(
                text=f"Datos: Kaggle CSV ({len(self.df_raw):,} registros)", fg="#4ade80"
            )
            self.ejecutar_pipeline_forecasting()
            messagebox.showinfo("Dataset Cargado", "El dataset de Kaggle se cargó y procesó correctamente.")

        except Exception as e:
            messagebox.showerror("Error al leer CSV", f"Ocurrió un error al cargar el archivo:\n{e}")

    def generar_datos_demo(self, mostrar_alerta=True):
        """Genera un dataset realista idéntico al de Kaggle (Store Item Demand) para pruebas inmediatas."""
        np.random.seed(42)
        fechas = pd.date_range(start="2023-01-01", end="2026-10-01", freq="D")
        n = len(fechas)
        registros = []

        t = np.arange(n)
        estacionalidad_anual = 12 * np.sin(2 * np.pi * fechas.dayofyear / 365.25)
        efecto_semana = np.where(fechas.dayofweek >= 5, 14, 0)  # Mayor venta en fin de semana

        for store in [1, 2, 3]:
            for item in [1, 2, 3, 4, 5]:
                base = 35 + (store * 8) + (item * 5)
                tendencia = 0.018 * t
                ruido = np.random.normal(0, 4.5, size=n)
                ventas = np.maximum(5, np.round(base + tendencia + estacionalidad_anual + efecto_semana + ruido))

                df_temp = pd.DataFrame({
                    'date': fechas,
                    'store': store,
                    'item': item,
                    'sales': ventas.astype(int)
                })
                registros.append(df_temp)

        self.df_raw = pd.concat(registros, ignore_index=True)
        self.combo_store['values'] = [1, 2, 3]
        self.combo_item['values'] = [1, 2, 3, 4, 5]
        self.combo_store.current(0)
        self.combo_item.current(0)

        self.lbl_estado_datos.config(
            text=f"Datos: Demo Sintético ({len(self.df_raw):,} filas)", fg="#38bdf8"
        )
        self.ejecutar_pipeline_forecasting()

        if mostrar_alerta:
            messagebox.showinfo(
                "Datos Demo Generados",
                "Se generó una serie de tiempo de 3 tiendas y 5 artículos con tendencia, "
                "estacionalidad anual y picos de fin de semana."
            )

    # =========================================================================
    # MOTOR DE INGENIERÍA DE CARACTERÍSTICAS Y FORECASTING (REGRESIÓN)
    # =========================================================================
    def _extraer_features_calendario(self, fecha, time_idx):
        dayofweek = fecha.dayofweek
        month = fecha.month
        dayofyear = fecha.timetuple().tm_yday
        is_weekend = 1 if dayofweek in (5, 6) else 0
        sin_doy = np.sin(2 * np.pi * dayofyear / 365.25)
        cos_doy = np.cos(2 * np.pi * dayofyear / 365.25)
        return dayofweek, month, is_weekend, sin_doy, cos_doy

    def preparar_dataset_supervisado(self, df_serie):
        df = df_serie.copy().sort_values('date').reset_index(drop=True)
        df['time_idx'] = np.arange(len(df))
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        dayofyear = df['date'].dt.dayofyear
        df['sin_dayofyear'] = np.sin(2 * np.pi * dayofyear / 365.25)
        df['cos_dayofyear'] = np.cos(2 * np.pi * dayofyear / 365.25)

        # Rezagos (Lags)
        df['lag_1'] = df['sales'].shift(1)
        df['lag_7'] = df['sales'].shift(7)
        df['lag_14'] = df['sales'].shift(14)
        df['lag_28'] = df['sales'].shift(28)

        # Ventanas Móviles (basadas en ventas hasta ayer para evitar fuga de datos)
        ventas_previas = df['sales'].shift(1)
        df['rolling_mean_7'] = ventas_previas.rolling(window=7).mean()
        df['rolling_mean_28'] = ventas_previas.rolling(window=28).mean()
        df['rolling_std_7'] = ventas_previas.rolling(window=7).std()

        return df.dropna().reset_index(drop=True)

    def _instanciar_modelo(self, nombre_modelo):
        if "Ridge" in nombre_modelo:
            return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        elif "Lasso" in nombre_modelo:
            return make_pipeline(StandardScaler(), Lasso(alpha=0.15, max_iter=5000))
        elif "Random Forest" in nombre_modelo:
            return RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1)
        else:
            return make_pipeline(StandardScaler(), LinearRegression())

    def ejecutar_pipeline_forecasting(self):
        if self.df_raw is None:
            return

        try:
            store_id = int(self.combo_store.get())
            item_id = int(self.combo_item.get())
            horizonte = int(self.spin_horizonte.get())
            stock_actual = float(self.entry_stock_actual.get())
        except ValueError:
            messagebox.showwarning("Datos inválidos", "Verifica que el horizonte y el stock actual sean números válidos.")
            return

        # Filtrar serie de tiempo
        df_item = self.df_raw[(self.df_raw['store'] == store_id) & (self.df_raw['item'] == item_id)].copy()
        if len(df_item) < 120:
            messagebox.showwarning(
                "Serie muy corta",
                "Se requieren al menos 120 días históricos para entrenar y validar el modelo con rezagos."
            )
            return

        df_sup = self.preparar_dataset_supervisado(df_item)

        # 1. División Train / Test (Últimos 90 días para evaluar precisión)
        dias_test = 90
        train_df = df_sup.iloc[:-dias_test]
        test_df = df_sup.iloc[-dias_test:]

        X_train, y_train = train_df[self.feature_cols], train_df['sales']
        X_test, y_test = test_df[self.feature_cols], test_df['sales']

        nombre_modelo = self.combo_modelo.get()
        modelo_eval = self._instanciar_modelo(nombre_modelo)
        modelo_eval.fit(X_train, y_train)
        pred_test = modelo_eval.predict(X_test)

        # Métricas de Evaluación
        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        mape = np.mean(np.abs((y_test - pred_test) / np.maximum(1, y_test))) * 100
        r2 = r2_score(y_test, pred_test)

        # 2. Re-entrenar con el 100% de la historia para pronosticar el Futuro Real
        modelo_final = self._instanciar_modelo(nombre_modelo)
        modelo_final.fit(df_sup[self.feature_cols], df_sup['sales'])

        # Pronóstico Autoregresivo Recursivo día por día
        historial_ventas = df_item.sort_values('date')['sales'].tolist()
        ultima_fecha = df_item['date'].max()
        ultimo_idx = int(df_sup['time_idx'].max())

        fechas_futuras = []
        predicciones_futuras = []

        for paso in range(1, horizonte + 1):
            fecha_futura = ultima_fecha + timedelta(days=paso)
            idx_futuro = ultimo_idx + paso

            dow, month, is_wknd, sin_doy, cos_doy = self._extraer_features_calendario(fecha_futura, idx_futuro)

            lag_1 = historial_ventas[-1]
            lag_7 = historial_ventas[-7]
            lag_14 = historial_ventas[-14]
            lag_28 = historial_ventas[-28]

            ultimos_7 = historial_ventas[-7:]
            ultimos_28 = historial_ventas[-28:]
            roll_mean_7 = float(np.mean(ultimos_7))
            roll_mean_28 = float(np.mean(ultimos_28))
            roll_std_7 = float(np.std(ultimos_7, ddof=1)) if len(ultimos_7) > 1 else 0.0

            fila_X = pd.DataFrame([[
                idx_futuro, dow, month, is_wknd, sin_doy, cos_doy,
                lag_1, lag_7, lag_14, lag_28,
                roll_mean_7, roll_mean_28, roll_std_7
            ]], columns=self.feature_cols)

            venta_pred = max(0.0, float(modelo_final.predict(fila_X)[0]))

            fechas_futuras.append(fecha_futura)
            predicciones_futuras.append(venta_pred)
            historial_ventas.append(venta_pred)  # Alimenta el siguiente paso recursivo

        # 3. Cálculos de Inventario y Stock de Seguridad
        confianza_str = self.combo_confianza.get()
        z_map = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
        z_val = z_map.get(confianza_str[:3], 1.65)

        pred_array = np.array(predicciones_futuras)
        demanda_total_pronosticada = float(np.sum(pred_array))
        # El error acumulado en H días escala con la raíz cuadrada del horizonte
        stock_seguridad = float(z_val * rmse * np.sqrt(horizonte))
        inventario_requerido = demanda_total_pronosticada + stock_seguridad
        pedido_sugerido = max(0.0, inventario_requerido - stock_actual)

        # Actualizar KPIs Superiores
        self.kpi_mae["val"].config(text=f"{mae:.2f} u.")
        self.kpi_mae["sub"].config(text=f"MAPE: {mape:.1f}% de variación")

        self.kpi_rmse["val"].config(text=f"{rmse:.2f} | R²: {r2:.2f}")
        self.kpi_rmse["sub"].config(text=nombre_modelo.split(" (")[0])

        self.kpi_demanda["val"].config(text=f"{ int(round(demanda_total_pronosticada)):,} u.")
        self.kpi_demanda["sub"].config(text=f"En los próximos {horizonte} días")

        self.kpi_safety["val"].config(text=f"+{int(round(stock_seguridad)):,} u.")
        self.kpi_safety["sub"].config(text=f"Nivel confianza {confianza_str[:3]}")

        if pedido_sugerido > 0:
            self.kpi_reorden["val"].config(text=f"PEDIR {int(np.ceil(pedido_sugerido)):,} u.", foreground="#dc2626")
            self.kpi_reorden["sub"].config(
                text=f"Requerido: {int(round(inventario_requerido)):,} vs Stock: {int(stock_actual):,}",
                fg="#dc2626"
            )
        else:
            sobrante = int(stock_actual - inventario_requerido)
            self.kpi_reorden["val"].config(text="STOCK ÓPTIMO", foreground="#16a34a")
            self.kpi_reorden["sub"].config(text=f"Superávit de {sobrante:,} unidades", fg="#16a34a")

        # 4. Actualizar Gráficas y Tabla
        self._dibujar_grafica_forecast(
            test_df['date'], y_test, pred_test,
            fechas_futuras, pred_array, rmse, z_val, store_id, item_id
        )
        self._dibujar_grafica_pesos(modelo_final, nombre_modelo)
        self._actualizar_tabla_pronostico(fechas_futuras, pred_array, rmse, z_val)

    def _dibujar_grafica_forecast(self, fechas_test, y_test, pred_test, fechas_fut, pred_fut, rmse, z_val, store, item):
        self.ax_forecast.clear()

        # Histórico de prueba (últimos 90 días)
        self.ax_forecast.plot(
            fechas_test, y_test, label="Ventas Reales (Últimos 90 días)",
            color="#94a3b8", linewidth=1.6, alpha=0.85
        )
        self.ax_forecast.plot(
            fechas_test, pred_test, label="Ajuste del Modelo (Backtest)",
            color="#0284c7", linestyle="--", linewidth=1.8
        )

        # Pronóstico futuro e intervalo de confianza
        self.ax_forecast.plot(
            fechas_fut, pred_fut, label=f"Pronóstico Futuro ({len(fechas_fut)} días)",
            color="#ea580c", linewidth=2.4
        )

        margen = z_val * rmse
        lim_inf = np.maximum(0, pred_fut - margen)
        lim_sup = pred_fut + margen
        self.ax_forecast.fill_between(
            fechas_fut, lim_inf, lim_sup, color="#fdba74", alpha=0.35,
            label="Intervalo de Confianza"
        )

        # Línea divisoria del día actual (inicio del pronóstico)
        self.ax_forecast.axvline(x=fechas_test.iloc[-1], color="#1e293b", linestyle=":", linewidth=1.5)

        self.ax_forecast.set_title(
            f"Pronóstico de Demanda — Tienda #{store} | Artículo #{item}",
            fontsize=12, fontweight="bold", pad=12
        )
        self.ax_forecast.set_xlabel("Fecha")
        self.ax_forecast.set_ylabel("Unidades Vendidas")
        self.ax_forecast.legend(loc="upper left", frameon=True)
        self.ax_forecast.grid(True, linestyle="--", alpha=0.35)
        self.fig_forecast.tight_layout()
        self.canvas_forecast.draw()

    def _dibujar_grafica_pesos(self, modelo_entrenado, nombre_modelo):
        self.ax_pesos.clear()

        nombres_amigables = [
            'Tendencia (Tiempo)', 'Día de Semana', 'Mes del Año', 'Fin de Semana',
            'Onda Anual (Sin)', 'Onda Anual (Cos)',
            'Venta Ayer (Lag 1)', 'Hace 1 Sem (Lag 7)', 'Hace 2 Sem (Lag 14)', 'Hace 4 Sem (Lag 28)',
            'Media Móvil 7d', 'Media Móvil 28d', 'Volatilidad 7d (Std)'
        ]

        if hasattr(modelo_entrenado, "feature_importances_"):
            importancias = modelo_entrenado.feature_importances_
            subtitulo = "Importancia Relativa de Variables (Gini / Reducción de Varianza)"
        else:
            regresor = modelo_entrenado.steps[-1][1]
            importancias = regresor.coef_
            subtitulo = "Coeficientes Estandarizados de la Regresión (Impacto Positivo / Negativo)"

        indices = np.argsort(np.abs(importancias))
        etiquetas_ord = [nombres_amigables[i] for i in indices]
        valores_ord = importancias[indices]
        colores = ["#2563eb" if v >= 0 else "#e11d48" for v in valores_ord]

        self.ax_pesos.barh(etiquetas_ord, valores_ord, color=colores, height=0.65)
        self.ax_pesos.axvline(0, color="#334155", linewidth=1)
        self.ax_pesos.set_title(f"{nombre_modelo} — {subtitulo}", fontsize=11, fontweight="bold", pad=12)
        self.ax_pesos.set_xlabel("Magnitud del Peso / Coeficiente")
        self.ax_pesos.grid(True, axis="x", linestyle="--", alpha=0.35)
        self.fig_pesos.tight_layout()
        self.canvas_pesos.draw()

    def _actualizar_tabla_pronostico(self, fechas_futuras, pred_array, rmse, z_val):
        for item in self.tree.get_children():
            self.tree.delete(item)

        dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        margen = z_val * rmse
        acumulado = 0.0
        registros_exportar = []

        for fecha, pred in zip(fechas_futuras, pred_array):
            venta_redondeada = int(round(pred))
            lim_inf = int(max(0, round(pred - margen)))
            lim_sup = int(round(pred + margen))
            acumulado += venta_redondeada
            nombre_dia = dias_es[fecha.weekday()]
            fecha_str = fecha.strftime("%Y-%m-%d")

            self.tree.insert("", tk.END, values=(
                fecha_str,
                nombre_dia,
                f"{venta_redondeada} u.",
                f"{lim_inf} u.",
                f"{lim_sup} u.",
                f"{int(acumulado):,} u."
            ))

            registros_exportar.append({
                "Fecha": fecha_str,
                "Dia_Semana": nombre_dia,
                "Venta_Pronosticada": venta_redondeada,
                "Limite_Inferior": lim_inf,
                "Limite_Superior": lim_sup,
                "Demanda_Acumulada": int(acumulado)
            })

        self.df_forecast_result = pd.DataFrame(registros_exportar)

    def exportar_pronostico_csv(self):
        if self.df_forecast_result is None or self.df_forecast_result.empty:
            messagebox.showwarning("Sin datos", "Primero ejecuta un pronóstico para poder exportarlo.")
            return

        ruta = filedialog.asksaveasfilename(
            title="Guardar Reporte de Pronóstico",
            defaultextension=".csv",
            initialfile="reporte_pronostico_inventario.csv",
            filetypes=[("Archivo CSV", "*.csv")]
        )
        if ruta:
            self.df_forecast_result.to_csv(ruta, index=False, encoding="utf-8-sig")
            messagebox.showinfo("Exportación Exitosa", f"El pronóstico se guardó correctamente en:\n{ruta}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaForecastingIA(root)
    root.mainloop()