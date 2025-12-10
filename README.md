# 📌 Prueba de Corridas (Runs Test) 

## Información del Sistema

**Requisitos:** numpy, scipy, matplotlib, pandas  
**Ejecutar:** `python main.py`

---
## Instalación y ejecución

### Requisitos del sistema

```bash
# Instalación de dependencias
pip install numpy scipy matplotlib pandas
```


### ¿Qué es la prueba de corridas?

La **prueba de corridas** es una prueba **no paramétrica** usada para evaluar si una secuencia de dos tipos (ej. Verdadero/Falso, 1/0, arriba/abajo) es compatible con **aleatoriedad**.

### Idea central

En una secuencia aleatoria:

- Muchas alternancias → secuencia "más variable".
- Muy pocas alternancias → bloques largos → posible patrón, tendencia o dependencia.

---

## Aplicaciones implementadas

El sistema implementa **3 aplicaciones** principales en pestañas separadas:

### 1. App1: Probabilidad de R (Exact Distribution of Runs)

**Nombre técnico:** Análisis de la Distribución Exacta del Número de Corridas

**Descripción:**  
Calcula probabilidades exactas usando análisis combinatorio cuando se conoce el número total de elementos de cada tipo (n₁ y n₂).

**Entrada:**
- `n1`: Número de elementos tipo S
- `n2`: Número de elementos tipo F  
- `R objetivo`: Número de corridas a analizar

**Salida:**
- P(R = r): Probabilidad exacta
- P(R ≤ r): Probabilidad acumulada
- Distribución completa de probabilidades
- Gráfico de barras con la distribución

**Uso:** Muestras pequeñas/medianas o cuando se requiere probabilidad exacta.

---

### 2. App2: Runs test (secuencia) - Wald-Wolfowitz Test

**Nombre técnico:** Prueba de Corridas para Aleatoriedad en Secuencias Binarias

**Descripción:**  
Aplica el test clásico de Wald-Wolfowitz usando aproximación normal para evaluar aleatoriedad en secuencias binarias observadas.

**Entrada:**
- Secuencia binaria (formatos soportados: T/F, S/F, 1/0, auto-detect)
- Nivel de significancia α (por defecto: 0.05)
- Opción de cargar desde archivo (txt/csv)

**Salida:**
- Corridas observadas R
- Estadísticos: μᴿ, σᴿ, Z, p-value
- Decisión estadística (rechazar/no rechazar H₀)
- Detalle de cada corrida (valor y longitud)
- Gráfico step plot de la secuencia

**Características:**
- Normalización automática a 0/1
- Detección automática de tipos de secuencia
- Análisis de longitudes de corridas (máx, mín, promedio)

---

### 3. App3: Control de Proceso - Above-Below Runs Test

**Nombre técnico:** Prueba de Corridas Arriba-Abajo Respecto al Promedio

**Descripción:**  
Convierte datos numéricos continuos a secuencia binaria comparando con un umbral (media, mediana o valor especificado), luego aplica el runs test para detectar falta de control en procesos.

**Entrada:**
- Valores numéricos (separados por espacios)
- Tipo de referencia: media, mediana, o valor especificado
- Opción de cargar CSV (columna numérica)

**Salida:**
- Análisis completo del runs test sobre signos
- Estadísticas descriptivas (media, mediana, desv. std, min, max)
- Gráfico de serie temporal con línea de referencia

**Aplicaciones:**
- Control estadístico de procesos
- Detección de tendencias en producción
- Monitoreo de calidad industrial
- Análisis de estabilidad de procesos

---

##  Algoritmos implementados

### 1. Distribución exacta (app1_calcular)

**Método:** Análisis combinatorio

Para cada R en 1, ..., n:

#### Si R es par (R = 2k):

$$\text{casos} = 2 \cdot \binom{n_1 - 1}{k - 1} \cdot \binom{n_2 - 1}{k - 1}$$

#### Si R es impar (R = 2k + 1):

$$\text{casos} = \binom{n_1 - 1}{k} \binom{n_2 - 1}{k - 1} + \binom{n_1 - 1}{k - 1} \binom{n_2 - 1}{k}$$

**Probabilidad:**

$$P(R) = \frac{\text{casos}}{\binom{n}{n_1}}$$

**Función clave:**
```python
def combos(n, k):
    """Combinación segura (enteros)."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)
```

---

### 2. Conteo de corridas (runs_test_sequence)

**Algoritmo:**
1. Recorre la secuencia
2. Inicializa `runs = 1`
3. Incrementa cada vez que el valor cambia

```python
runs = 1
for i in range(1, n):
    if seq_num[i] != seq_num[i-1]:
        runs += 1
```

**Salida adicional:**
- Detalle de cada corrida (valor, longitud, posición inicial)
- Resumen de longitudes (máx, mín, promedio)

---

### 3. Runs test con aproximación normal

**Estadísticos calculados:**

- **Corridas observadas:** R

- **Media esperada:**

$$\mu_R = \frac{2n_1 n_2}{n} + 1$$

- **Varianza:**

$$\sigma^2_R = \frac{2n_1 n_2 (2n_1 n_2 - n)}{n^2(n-1)}$$

- **Estadístico Z:**

$$Z = \frac{R - \mu_R}{\sigma_R}$$

- **p-value bilateral:**

$$p = 2 \left[1 - \Phi(|Z|)\right]$$

**Implementación:**
```python
mu = (2.0 * n1 * n2) / n + 1.0
var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n**2 * (n - 1))
sigma = math.sqrt(var) if var > 0 else 0.0
Z = (runs - mu) / sigma if sigma > 0 else 0.0
p_value = 2.0 * (1.0 - norm.cdf(abs(Z)))
```

---

### 4. Control de proceso (runs_test_numeric)

**Proceso:**
1. Calcula el umbral de referencia (media, mediana o valor dado)
2. Convierte valores a signos binarios: `1` si valor > referencia, `0` si valor ≤ referencia
3. Aplica `runs_test_sequence` sobre los signos
4. Calcula estadísticas descriptivas adicionales

```python
signs = [1 if v > reference else 0 for v in values]
```

---

## Interpretación de resultados

### Si p-value > α

**Decisión:** No se rechaza H₀  
**Conclusión:** La secuencia es **compatible con aleatoriedad**

**Mensaje del sistema:**
```
"DECISIÓN: No se rechaza H0 → Secuencia compatible con aleatoriedad."
```

---

###  Si p-value ≤ α

**Decisión:** Se rechaza H₀  
**Conclusión:** Hay evidencia de **no aleatoriedad** (patrón, tendencia, periodicidad)

**Mensaje del sistema:**
```
"DECISIÓN: Se rechaza H0 → Secuencia no aleatoria (evidencia de patrón)."
```

---

### Interpretación de distribución exacta (App1)

**Reglas heurísticas implementadas:**

- Si P(R ≤ r) < 0.025:  
  → "Probabilidad acumulada muy baja → Observación inusual"

- Si P(R ≤ r) > 0.975:  
  → "Probabilidad acumulada muy alta → Observación inusual en sentido opuesto"

- De lo contrario:  
  → "Observación compatible con la hipótesis de aleatoriedad"

---

## Gráficas producidas por la aplicación

### 1. Distribución P(R) – App1

**Tipo:** Gráfico de barras  
**Función:** `_draw_app1_graph(probs, R_obj)`

**Elementos:**
- Eje X: Número de corridas R
- Eje Y: Probabilidad P(R)
- Línea vertical roja punteada: R objetivo
- Título: "Distribución exacta de corridas"

**Utilidad:**
- Ver la distribución completa de probabilidades
- Identificar si el R observado cae en las colas
- Visualizar la forma de la distribución

---

### 2. Secuencia binaria (step plot) – App2

**Tipo:** Step plot  
**Función:** `_draw_app2_sequence(seq_num)`

**Elementos:**
- Eje X: Posición en la secuencia (1 a n)
- Eje Y: Valor binario (0 o 1)
- Estilo: step plot con `where='mid'`
- Título: "Secuencia binaria (0/1)"

**Utilidad:**
- Visualizar patrones de agrupamiento
- Identificar alternancias frecuentes
- Detectar bloques largos de un mismo valor
- Observar la estructura temporal de la secuencia

---

### 3. Serie temporal con línea de referencia – App3

**Tipo:** Gráfico de línea con marcadores  
**Función:** `_draw_app3(values, ref)`

**Elementos:**
- Línea con marcadores circulares (valores observados)
- Línea horizontal roja punteada: umbral de referencia
- Eje X: Número de observación
- Eje Y: Valor medido
- Leyenda: valor de referencia
- Título: "Serie temporal del proceso"

**Utilidad:**
- Observar tendencias en el tiempo
- Identificar ciclos o periodicidad
- Ver cruces frecuentes/infrecuentes del umbral
- Evaluar estabilidad del proceso

---

## Características de carga de datos

### App2: Carga de archivos para secuencias

**Función:** `app2_load_file()`

**Formatos soportados:**
- Archivos de texto (.txt)
- CSV (.csv)
- Archivos de datos (.dat)

**Proceso:**
1. Intenta leer como CSV con pandas
2. Extrae todos los tokens (aplana filas y columnas)
3. Si falla, lee como texto plano
4. Inserta tokens en el campo de entrada

**Código:**
```python
def app2_load_file(self):
    fn = filedialog.askopenfilename(
        filetypes=[("Text/CSV","*.txt;*.csv;*.dat"),("All","*.*")]
    )
    if not fn:
        return
    try:
        df = pd.read_csv(fn, header=None)
        tokens = []
        for _, row in df.iterrows():
            for v in row:
                if pd.isna(v): continue
                tokens.extend(str(v).split())
        if not tokens:
            raise ValueError("Archivo sin tokens")
        self.app2_seq.delete('1.0', tk.END)
        self.app2_seq.insert('1.0', " ".join(tokens))
    except Exception:
        with open(fn, 'r', encoding='utf8', errors='ignore') as f:
            content = f.read().strip()
        self.app2_seq.delete('1.0', tk.END)
        self.app2_seq.insert('1.0', content)
```

---

### App3: Carga de CSV para datos numéricos

**Función:** `app3_load_csv()`

**Proceso:**
1. Lee archivo CSV con pandas
2. Identifica columnas numéricas automáticamente
3. Selecciona la primera columna numérica
4. Elimina valores NaN
5. Convierte a lista de floats

**Código:**
```python
def app3_load_csv(self):
    fn = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("All","*.*")])
    if not fn: return
    try:
        df = pd.read_csv(fn)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            messagebox.showerror("Error", 
                "No se encontraron columnas numéricas en el CSV")
            return
        col = numeric_cols[0]
        vals = df[col].dropna().astype(float).tolist()
        self.app3_values.delete('1.0', tk.END)
        self.app3_values.insert('1.0', " ".join(map(str, vals)))
    except Exception as e:
        messagebox.showerror("Error leyendo CSV", str(e))
```

---

## Sistema de exportación

### Pestaña de Análisis/Export

**Ubicación:** Cuarta pestaña del notebook  
**Función:** `_build_reports()`

**Funcionalidad:**
- Botones para exportar resultados de cada aplicación
- Exporta a archivos .txt con codificación UTF-8
- Mantiene formato completo de los resultados

**Implementación:**
```python
def _export_text(self, text):
    if not text:
        messagebox.showinfo("Exportar", "No hay texto para exportar")
        return
    fn = filedialog.asksaveasfilename(
        defaultextension=".txt", 
        filetypes=[("Text","*.txt")]
    )
    if not fn:
        return
    with open(fn, 'w', encoding='utf8') as f:
        f.write(text)
    messagebox.showinfo("Exportar", f"Exportado a {fn}")
```

**Ventajas:**
- Preserva todos los resultados para informes
- Formato legible para documentación
- Compatible con procesadores de texto

---

## Normalización de secuencias (App2)

### Tipos de secuencia soportados

**1. T/F (True/False):**
```python
mapping = {'T':'1', 'F':'0', 'TRUE':'1', 'FALSE':'0', 
           'V':'1', 'FALSO':'0'}
seq = [mapping.get(t.upper(), t) for t in tokens]
```

**2. S/F (Success/Failure):**
```python
mapping = {'S':'1', 'F':'0'}
seq = [mapping.get(t.upper(), t) for t in tokens]
```

**3. 1/0 (Binario directo):**
```python
seq = [t if t in ('1','0') else t for t in tokens]
```

**4. Auto-detect:**
```python
uniq = sorted(set(tokens), key=lambda x: tokens.index(x))
if len(uniq) != 2:
    # Error: necesita exactamente 2 valores
seq = ['1' if t == uniq[0] else '0' for t in tokens]
```

---

## Análisis detallado de corridas (App2)

### Información de cada corrida

El sistema identifica y analiza cada corrida individualmente:

```python
runs_list = []
current = seq_num[0]
length = 1
starts = [0]
lengths = []
types = []

for i in range(1, n):
    if seq_num[i] == current:
        length += 1
    else:
        runs_list.append((current, length))
        types.append(current)
        lengths.append(length)
        current = seq_num[i]
        length = 1
        starts.append(i)
runs_list.append((current, length))
```

**Salida generada:**
```
Detalle de corridas:
  Corrida 1: valor=1, longitud=2
  Corrida 2: valor=0, longitud=3
  Corrida 3: valor=1, longitud=1
  ...

Resumen longitudes: max=5, min=1, avg=2.50
```

---

## Validaciones y manejo de errores

### Validación de entrada (App1)

```python
try:
    n1 = int(self.app1_n1.get())
    n2 = int(self.app1_n2.get())
    R_obj = int(self.app1_R.get())
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 y n2 deben ser positivos.")
except Exception as e:
    messagebox.showerror("Entrada inválida", str(e))
    return
```

### Validación de secuencia binaria (App2)

```python
if not all(s in ('0','1') for s in seq):
    messagebox.showerror("Error", 
        "La secuencia debe convertirse a 0/1 (revisar tipo seleccionado)")
    return
```

### Manejo de casos especiales

**Secuencia vacía:**
```python
if n == 0:
    return ("Secuencia vacía", seq_num)
```

**Valores todos iguales:**
```python
if n1 > 0 and n2 > 0 and n > 1:
    # Calcular estadísticos normalmente
else:
    mu = sigma = Z = 0.0
    p_value = 1.0
```

---

## Estructura del programa

### Clase principal: RunsApp

```python
class RunsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prueba de Corridas - Aplicaciones")
        self.root.geometry("1200x800")
        
        # Notebook con 4 pestañas
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill='both', expand=True, padx=8, pady=8)
        
        self._build_app1()  # Distribución exacta
        self._build_app2()  # Runs test binario
        self._build_app3()  # Control de proceso
        self._build_reports()  # Exportación
```

### Métodos principales

| Método | Descripción |
|--------|-------------|
| `_build_app1()` | Construye interfaz App1 |
| `app1_calcular()` | Calcula probabilidades exactas |
| `_draw_app1_graph()` | Dibuja gráfico de barras |
| `_build_app2()` | Construye interfaz App2 |
| `app2_run()` | Ejecuta test de Wald-Wolfowitz |
| `runs_test_sequence()` | Algoritmo principal de runs test |
| `_draw_app2_sequence()` | Dibuja step plot |
| `_build_app3()` | Construye interfaz App3 |
| `app3_run()` | Ejecuta test arriba-abajo |
| `runs_test_numeric()` | Convierte numérico a binario |
| `_draw_app3()` | Dibuja serie temporal |
| `_export_text()` | Exporta resultados |

---

## Fórmulas de referencia rápida

| Concepto | Fórmula |
|----------|---------|
| Media de corridas | $\mu_R = \dfrac{2n_1 n_2}{n} + 1$ |
| Varianza de corridas | $\sigma^2_R = \dfrac{2n_1 n_2 (2n_1 n_2 - n)}{n^2(n-1)}$ |
| Estadístico Z | $Z = \dfrac{R - \mu_R}{\sigma_R}$ |
| p-value bilateral | $p = 2[1 - \Phi(\|Z\|)]$ |
| Combinatoria total | $\dbinom{n}{n_1}$ |
| Corridas pares | $2 \cdot \dbinom{n_1-1}{k-1} \cdot \dbinom{n_2-1}{k-1}$ |
| Corridas impares | $\dbinom{n_1-1}{k}\dbinom{n_2-1}{k-1} + \dbinom{n_1-1}{k-1}\dbinom{n_2-1}{k}$ |

---

## Ejemplos de uso

### Ejemplo 1: Probabilidad exacta (App1)

**Entrada:**
- n1 = 5 (elementos S)
- n2 = 3 (elementos F)
- R objetivo = 3

**Proceso:**
1. Total de arreglos = C(8,5) = 56
2. Para R = 3 (impar, k = 1):
   - casos = C(4,1)×C(2,0) + C(4,0)×C(2,1) = 4×1 + 1×2 = 6
3. P(R = 3) = 6/56 = 0.107143

**Salida:**
```
P(R = 3) = 0.107143
P(R ≤ 3) = 0.142857
Interpretación: Observación compatible con aleatoriedad
```

---

### Ejemplo 2: Test de secuencia (App2)

**Entrada:**
```
T F F T F T F T T F T F F T F T F T T F
```

**Proceso:**
1. n = 20, n1 = 10, n2 = 10
2. Corridas observadas R = 18
3. μR = (2×10×10)/20 + 1 = 11.00
4. σR = 2.179
5. Z = (18 - 11)/2.179 = 3.211
6. p-value = 0.0013

**Salida:**
```
DECISIÓN: Se rechaza H0 → Secuencia no aleatoria
Diagnóstico: Demasiadas corridas (patrón alternante)
```

---

### Ejemplo 3: Control de proceso (App3)

**Entrada:**
```
10.2 9.8 10.5 9.9 10.3 10.1 9.7 10.4 10.0
```

**Referencia:** media = 10.1

**Proceso:**
1. Signos: + - + - + 0 - + -
2. n = 9, n1 = 4, n2 = 5
3. R = 8 (muchas alternancias)
4. μR = 5.444, σR = 1.528
5. Z = 1.673, p-value = 0.094

**Salida:**
```
DECISIÓN: No se rechaza H0 (α = 0.05)
El proceso está bajo control estadístico
```








