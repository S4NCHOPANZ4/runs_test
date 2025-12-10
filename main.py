# runs_app.py
# Refactorización del programa de Pruebas de Corridas (3 aplicaciones)
# - App1: Probabilidades exactas de R corridas (combinatoria correcta)
# - App2: Runs test para secuencias binarias (Z y p-value) - MEJORADA CON GRÁFICA DE DOS COLAS
# - App3: Runs test aplicado a datos numéricos frente a umbral (media/mediana/específico)
#
# Requisitos: numpy, scipy, matplotlib, pandas (opcional para CSV)
# Ejecutar: python runs_app.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import math
from scipy.stats import norm
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')


def combos(n, k):
    """Combinación segura (enteros)."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


class RunsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prueba de Corridas - Aplicaciones")
        self.root.geometry("1200x800")

        # Notebook con 4 pestañas (3 apps + análisis/export)
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill='both', expand=True, padx=8, pady=8)

        self._build_app1()
        self._build_app2()
        self._build_app3()
        self._build_reports()

    # -----------------------------
    # APP 1: Probabilidad exacta
    # -----------------------------
    def _build_app1(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text="App1: Probabilidad de R")

        left = ttk.Frame(frame)
        left.pack(side='left', fill='both', expand=True, padx=6, pady=6)

        right = ttk.Frame(frame)
        right.pack(side='right', fill='both', expand=True, padx=6, pady=6)

        inp = ttk.LabelFrame(left, text="Parámetros")
        inp.pack(fill='x', padx=6, pady=6)

        ttk.Label(inp, text="n1 (S):").grid(row=0, column=0, sticky='w', padx=4, pady=4)
        self.app1_n1 = ttk.Entry(inp, width=10); self.app1_n1.insert(0, "5"); self.app1_n1.grid(row=0, column=1)

        ttk.Label(inp, text="n2 (F):").grid(row=1, column=0, sticky='w', padx=4, pady=4)
        self.app1_n2 = ttk.Entry(inp, width=10); self.app1_n2.insert(0, "3"); self.app1_n2.grid(row=1, column=1)

        ttk.Label(inp, text="R objetivo:").grid(row=2, column=0, sticky='w', padx=4, pady=4)
        self.app1_R = ttk.Entry(inp, width=10); self.app1_R.insert(0, "3"); self.app1_R.grid(row=2, column=1)

        btnf = ttk.Frame(inp); btnf.grid(row=3, column=0, columnspan=2, pady=8)
        ttk.Button(btnf, text="Calcular", command=self.app1_calcular).pack(side='left', padx=4)
        ttk.Button(btnf, text="Exportar texto", command=lambda: self._export_text(self.app1_output)).pack(side='left')
        ttk.Button(btnf, text="Exportar PNG", command=lambda: self._export_png(self.app1_fig)).pack(side='left', padx=4)

        outf = ttk.LabelFrame(left, text="Resultados (texto)")
        outf.pack(fill='both', expand=True, padx=6, pady=6)
        self.app1_text = scrolledtext.ScrolledText(outf, height=18)
        self.app1_text.pack(fill='both', expand=True)

        # gráfico simple (distribución de P(R))
        graphf = ttk.LabelFrame(right, text="Distribución P(R)")
        graphf.pack(fill='both', expand=True, padx=6, pady=6)
        self.app1_fig = Figure(figsize=(5,4), dpi=100)
        self.app1_canvas = FigureCanvasTkAgg(self.app1_fig, master=graphf)
        self.app1_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.app1_output = ""

    def app1_calcular(self):
        try:
            n1 = int(self.app1_n1.get())
            n2 = int(self.app1_n2.get())
            R_obj = int(self.app1_R.get())
            if n1 <= 0 or n2 <= 0:
                raise ValueError("n1 y n2 deben ser positivos.")
        except Exception as e:
            messagebox.showerror("Entrada inválida", str(e))
            return

        n = n1 + n2
        total = combos(n, n1)
        probs = {}  # R -> probability

        # Fórmulas combinatorias (correctas):
        # Si R = 2k (par): número = 2 * C(n1-1, k-1) * C(n2-1, k-1)
        # Si R = 2k+1 (impar): número = C(n1-1, k) * C(n2-1, k-1) + C(n1-1, k-1) * C(n2-1, k)
        for R in range(1, n+1):
            if R < 1:
                continue
            if R % 2 == 0:
                k = R // 2
                ways = 2 * combos(n1-1, k-1) * combos(n2-1, k-1)
            else:
                k = R // 2  # R=2k+1 -> k = (R-1)/2
                # two possible distributions of runs: (k+1,k) and (k,k+1)
                ways = combos(n1-1, k) * combos(n2-1, k-1) + combos(n1-1, k-1) * combos(n2-1, k)
            if ways > 0:
                probs[R] = ways / total

        # Build output
        lines = []
        lines.append("APLICACIÓN 1: PROBABILIDAD DE CORRIDAS (exacta)\n")
        lines.append(f"n1 = {n1}, n2 = {n2}, n = {n}\n")
        lines.append(f"Total combinaciones = C({n},{n1}) = {total}\n")
        lines.append("-"*60 + "\n")

        p_eq = probs.get(R_obj, 0.0)
        p_le = sum(p for r,p in probs.items() if r <= R_obj)

        lines.append(f"P(R = {R_obj}) = {p_eq:.6f}\n")
        lines.append(f"P(R ≤ {R_obj}) = {p_le:.6f}\n\n")
        lines.append("Distribución completa (R : P(R)):\n")
        for r in sorted(probs.keys()):
            lines.append(f"  R={r:2d} : {probs[r]:.6f}\n")

        # Interpretation (simple)
        lines.append("\nINTERPRETACIÓN (reglas heurísticas):\n")
        if p_le < 0.025:
            lines.append("  Probabilidad acumulada muy baja → Observación inusual (poca probabilidad)\n")
        elif p_le > 0.975:
            lines.append("  Probabilidad acumulada muy alta → Observación inusual en sentido opuesto\n")
        else:
            lines.append("  Observación compatible con la hipótesis de aleatoriedad\n")

        self.app1_output = "".join(lines)
        self.app1_text.delete('1.0', tk.END)
        self.app1_text.insert(tk.END, self.app1_output)

        # Dibujar gráfico simple
        self._draw_app1_graph(probs, R_obj)

    def _draw_app1_graph(self, probs, R_obj):
        self.app1_fig.clear()
        ax = self.app1_fig.add_subplot(111)
        R_vals = sorted(probs.keys())
        P_vals = [probs[r] for r in R_vals]
        ax.bar(R_vals, P_vals)
        ax.set_xlabel("Número de corridas R")
        ax.set_ylabel("P(R)")
        ax.set_title("Distribución exacta de corridas")
        if R_obj in R_vals:
            ax.axvline(R_obj, color='red', linestyle='--', label=f'R objetivo={R_obj}')
            ax.legend()
        self.app1_canvas.draw()

    # -----------------------------
    # APP 2: Prueba de corridas para secuencias binarias - MEJORADA
    # -----------------------------
    def _build_app2(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text="App2: Runs test (secuencia)")

        top = ttk.LabelFrame(frame, text="Entrada")
        top.pack(fill='x', padx=6, pady=6)

        ttk.Label(top, text="Secuencia (ej: T F F T ... o 1 0 1 1):").grid(row=0, column=0, sticky='w')
        self.app2_seq = scrolledtext.ScrolledText(top, height=3)
        self.app2_seq.insert('1.0', "T F F T F T F T T F T F F T F T F T T F")
        self.app2_seq.grid(row=1, column=0, columnspan=4, sticky='ew', padx=4, pady=4)

        ttk.Label(top, text="Tipo:").grid(row=2, column=0, sticky='w', padx=4)
        self.app2_type = ttk.Combobox(top, values=["T/F", "S/F", "1/0", "auto"], width=8)
        self.app2_type.set("T/F")
        self.app2_type.grid(row=2, column=1)

        ttk.Label(top, text="α:").grid(row=2, column=2, sticky='e')
        self.app2_alpha = ttk.Entry(top, width=8); self.app2_alpha.insert(0, "0.05"); self.app2_alpha.grid(row=2, column=3, padx=4)

        btnf = ttk.Frame(top); btnf.grid(row=3, column=0, columnspan=4, pady=6)
        ttk.Button(btnf, text="Cargar archivo (txt/csv)", command=self.app2_load_file).pack(side='left', padx=4)
        ttk.Button(btnf, text="Ejecutar test", command=self.app2_run).pack(side='left', padx=4)
        ttk.Button(btnf, text="Exportar texto", command=lambda: self._export_text(self.app2_output)).pack(side='left')
        ttk.Button(btnf, text="Exportar PNG", command=lambda: self._export_png(self.app2_fig)).pack(side='left', padx=4)

        outf = ttk.LabelFrame(frame, text="Resultados")
        outf.pack(fill='both', expand=True, padx=6, pady=6)
        self.app2_text = scrolledtext.ScrolledText(outf, height=10)
        self.app2_text.pack(fill='x', padx=6, pady=6)

        # Gráficos mejorados (2 subplots)
        self.app2_fig = Figure(figsize=(10,5), dpi=100)
        self.app2_canvas = FigureCanvasTkAgg(self.app2_fig, master=outf)
        self.app2_canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)

        self.app2_output = ""
        self.app2_stats = {}  # Para almacenar estadísticas

    def app2_load_file(self):
        fn = filedialog.askopenfilename(filetypes=[("Text/CSV","*.txt;*.csv;*.dat"),("All","*.*")])
        if not fn:
            return
        try:
            # Try to read a single-line sequence, else raw text
            df = pd.read_csv(fn, header=None)
            # flatten all tokens
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

    def app2_run(self):
        seq_raw = self.app2_seq.get('1.0', tk.END).strip()
        if not seq_raw:
            messagebox.showerror("Error", "Ingrese una secuencia")
            return

        tokens = seq_raw.replace(',', ' ').replace('\n', ' ').split()
        typ = self.app2_type.get()
        # normalize to '1' and '0'
        if typ == "T/F":
            mapping = {'T':'1','F':'0','TRUE':'1','FALSE':'0','V':'1','FALSO':'0'}
            seq = [mapping.get(t.upper(), t) for t in tokens]
        elif typ == "S/F":
            mapping = {'S':'1','F':'0'}
            seq = [mapping.get(t.upper(), t) for t in tokens]
        elif typ == "1/0":
            seq = [t if t in ('1','0') else t for t in tokens]
        else:
            # auto try: pick two unique tokens
            uniq = sorted(set(tokens), key=lambda x: tokens.index(x))
            if len(uniq) != 2:
                messagebox.showerror("Error", "No se pudo auto-detectar: la secuencia necesita exactamente 2 valores distintos")
                return
            seq = [ '1' if t == uniq[0] else '0' for t in tokens ]

        # validate binary
        if not all(s in ('0','1') for s in seq):
            messagebox.showerror("Error", "La secuencia debe convertirse a 0/1 (revisar tipo seleccionado)")
            return

        try:
            alpha = float(self.app2_alpha.get())
        except:
            alpha = 0.05

        text, seq_plot, stats = self.runs_test_sequence(seq, alpha)
        self.app2_output = text
        self.app2_stats = stats
        self.app2_text.delete('1.0', tk.END)
        self.app2_text.insert(tk.END, text)
        self._draw_app2_improved(seq_plot, stats, alpha)

    def runs_test_sequence(self, seq, alpha=0.05):
        """Devuelve (texto_resultado, numeric_sequence para plot, stats_dict)."""
        seq_num = [int(x) for x in seq]
        n = len(seq_num)
        n1 = sum(seq_num)
        n2 = n - n1

        # Count runs
        if n == 0:
            return ("Secuencia vacía", seq_num, {})
        runs = 1
        for i in range(1, n):
            if seq_num[i] != seq_num[i-1]:
                runs += 1

        # Expectation and variance (approx normal)
        if n1 > 0 and n2 > 0 and n > 1:
            mu = (2.0 * n1 * n2) / n + 1.0
            var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n**2 * (n - 1))
            sigma = math.sqrt(var) if var > 0 else 0.0
            Z = (runs - mu) / sigma if sigma > 0 else 0.0
            p_value = 2.0 * (1.0 - norm.cdf(abs(Z))) if sigma > 0 else 1.0
        else:
            mu = sigma = Z = 0.0
            p_value = 1.0

        # Guardar estadísticas
        stats = {
            'n': n, 'n1': n1, 'n2': n2,
            'runs': runs, 'mu': mu, 'sigma': sigma,
            'Z': Z, 'p_value': p_value, 'alpha': alpha
        }

        lines = []
        lines.append("APLICACIÓN 2: RUNS TEST (SECUENCIA BINARIA)\n")
        lines.append(f"n = {n}, n1 = {n1}, n2 = {n2}\n")
        lines.append(f"Corridas observadas R = {runs}\n")
        lines.append(f"Esperado μ_R = {mu:.4f}, σ_R = {sigma:.4f}\n")
        lines.append(f"Z = {Z:.4f}, p-value (bilateral) = {p_value:.6f}\n")
        lines.append(f"Nivel α = {alpha}\n")
        
        # Valores críticos
        z_crit = norm.ppf(1 - alpha/2)
        lines.append(f"Valores críticos: Z_α/2 = ±{z_crit:.4f}\n\n")
        
        if p_value > alpha:
            lines.append("DECISIÓN: No se rechaza H0 → Secuencia compatible con aleatoriedad.\n")
        else:
            lines.append("DECISIÓN: Se rechaza H0 → Secuencia no aleatoria (evidencia de patrón).\n")

        # Additional diagnostics: lengths of runs
        runs_list = []
        current = seq_num[0]
        length = 1
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
        runs_list.append((current, length))
        types.append(current)
        lengths.append(length)

        lines.append("\nDetalle de corridas:\n")
        for idx, (val, lng) in enumerate(runs_list, start=1):
            lines.append(f"  Corrida {idx}: valor={val}, longitud={lng}\n")
        lines.append("\nResumen longitudes: max=%d, min=%d, avg=%.2f\n" % (max(lengths), min(lengths), sum(lengths)/len(lengths)))

        return ("".join(lines), seq_num, stats)

    def _draw_app2_improved(self, seq_num, stats, alpha):
        """Dibuja dos gráficos: secuencia y distribución normal con prueba de dos colas."""
        self.app2_fig.clear()
        
        # Subplot 1: Secuencia binaria
        ax1 = self.app2_fig.add_subplot(1, 2, 1)
        x = list(range(1, len(seq_num)+1))
        ax1.step(x, seq_num, where='mid', linewidth=1.5, color='steelblue')
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_yticks([0,1])
        ax1.set_yticklabels(['0','1'])
        ax1.set_xlabel("Posición")
        ax1.set_ylabel("Valor")
        ax1.set_title("Secuencia binaria observada")
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Distribución normal con prueba de dos colas
        ax2 = self.app2_fig.add_subplot(1, 2, 2)
        
        Z_obs = stats.get('Z', 0)
        z_crit = norm.ppf(1 - alpha/2)
        
        # Rango de valores Z para graficar
        z_range = np.linspace(-4, 4, 500)
        pdf_vals = norm.pdf(z_range)
        
        # Graficar la distribución normal
        ax2.plot(z_range, pdf_vals, 'k-', linewidth=2, label='N(0,1)')
        
        # Sombrear regiones críticas (dos colas)
        # Cola izquierda
        z_left = z_range[z_range <= -z_crit]
        pdf_left = norm.pdf(z_left)
        ax2.fill_between(z_left, 0, pdf_left, alpha=0.3, color='red', 
                         label=f'Región crítica α/2={alpha/2:.3f}')
        
        # Cola derecha
        z_right = z_range[z_range >= z_crit]
        pdf_right = norm.pdf(z_right)
        ax2.fill_between(z_right, 0, pdf_right, alpha=0.3, color='red')
        
        # Región de no rechazo
        z_middle = z_range[(z_range > -z_crit) & (z_range < z_crit)]
        pdf_middle = norm.pdf(z_middle)
        ax2.fill_between(z_middle, 0, pdf_middle, alpha=0.2, color='green',
                        label='Región de no rechazo')
        
        # Líneas verticales para valores críticos
        ax2.axvline(-z_crit, color='orange', linestyle='--', linewidth=1.5,
                   label=f'Z crítico = ±{z_crit:.3f}')
        ax2.axvline(z_crit, color='orange', linestyle='--', linewidth=1.5)
        
        # Valor Z observado
        ax2.axvline(Z_obs, color='blue', linestyle='-', linewidth=2,
                   label=f'Z obs = {Z_obs:.3f}')
        
        # Etiquetas y título
        ax2.set_xlabel('Valor Z')
        ax2.set_ylabel('Densidad de probabilidad')
        ax2.set_title(f'Prueba de dos colas (α={alpha})')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Añadir texto con el p-value
        p_val = stats.get('p_value', 1)
        decision_color = 'red' if p_val < alpha else 'green'
        ax2.text(0.02, 0.98, f'p-value = {p_val:.4f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=decision_color, alpha=0.3),
                fontsize=10, fontweight='bold')
        
        self.app2_fig.tight_layout()
        self.app2_canvas.draw()

    # -----------------------------
    # APP 3: Control de proceso con valores numéricos
    # -----------------------------
    def _build_app3(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text="App3: Control de Proceso")

        top = ttk.LabelFrame(frame, text="Entrada de datos")
        top.pack(fill='x', padx=6, pady=6)

        ttk.Label(top, text="Valores (separados por espacios o cargar CSV):").grid(row=0, column=0, sticky='w')
        self.app3_values = scrolledtext.ScrolledText(top, height=3)
        self.app3_values.insert('1.0', "10.2 9.8 10.5 9.9 10.3 10.1 9.7 10.4 10.0")
        self.app3_values.grid(row=1, column=0, columnspan=4, sticky='ew', padx=4, pady=4)

        ttk.Button(top, text="Cargar CSV (columna)", command=self.app3_load_csv).grid(row=2, column=0, padx=4, pady=4)
        ttk.Label(top, text="Referencia:").grid(row=2, column=1, padx=4)
        self.app3_ref_type = ttk.Combobox(top, values=["media","mediana","valor"], width=8)
        self.app3_ref_type.set("media")
        self.app3_ref_type.grid(row=2, column=2)
        self.app3_ref_value = ttk.Entry(top, width=10); self.app3_ref_value.insert(0, "10.0"); self.app3_ref_value.grid(row=2, column=3, padx=4)

        btnf = ttk.Frame(top); btnf.grid(row=3, column=0, columnspan=4, pady=6)
        ttk.Button(btnf, text="Analizar", command=self.app3_run).pack(side='left', padx=4)
        ttk.Button(btnf, text="Exportar texto", command=lambda: self._export_text(self.app3_output)).pack(side='left')

        outf = ttk.LabelFrame(frame, text="Resultados y gráficos")
        outf.pack(fill='both', expand=True, padx=6, pady=6)

        self.app3_text = scrolledtext.ScrolledText(outf, height=12)
        self.app3_text.pack(fill='x', padx=6, pady=6)

        self.app3_fig = Figure(figsize=(6,3), dpi=100)
        self.app3_canvas = FigureCanvasTkAgg(self.app3_fig, master=outf)
        self.app3_canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)

        self.app3_output = ""

    def app3_load_csv(self):
        fn = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("All","*.*")])
        if not fn: return
        try:
            df = pd.read_csv(fn)
            # If one column, take it. Else ask user to pick first numeric column.
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                messagebox.showerror("Error", "No se encontraron columnas numéricas en el CSV")
                return
            col = numeric_cols[0]
            vals = df[col].dropna().astype(float).tolist()
            self.app3_values.delete('1.0', tk.END)
            self.app3_values.insert('1.0', " ".join(map(str, vals)))
        except Exception as e:
            messagebox.showerror("Error leyendo CSV", str(e))

    def app3_run(self):
        txt = self.app3_values.get('1.0', tk.END).strip()
        if not txt:
            messagebox.showerror("Error", "Ingrese datos")
            return
        try:
            values = [float(x) for x in txt.replace(',', ' ').split()]
        except Exception as e:
            messagebox.showerror("Error parseando valores", str(e))
            return

        ref_type = self.app3_ref_type.get()
        if ref_type == "media":
            ref = float(np.mean(values))
        elif ref_type == "mediana":
            ref = float(np.median(values))
        else:
            try:
                ref = float(self.app3_ref_value.get())
            except:
                ref = float(np.mean(values))

        text = self.runs_test_numeric(values, ref)
        self.app3_output = text
        self.app3_text.delete('1.0', tk.END)
        self.app3_text.insert(tk.END, text)
        self._draw_app3(values, ref)

    def runs_test_numeric(self, values, reference):
        n = len(values)
        signs = [1 if v > reference else 0 for v in values]
        # apply the binary runs test to 'signs'
        summary, _, _ = self.runs_test_sequence([str(x) for x in signs], alpha=0.05)
        # Add numeric summary
        lines = []
        lines.append("APLICACIÓN 3: CONTROL DE PROCESO CON RUNS\n")
        lines.append(f"N = {n}\n")
        lines.append(f"Umbral de referencia = {reference}\n\n")
        lines.append(summary)
        # add descriptive stats
        arr = np.array(values)
        lines.append("\nEstadísticas descriptivas:\n")
        lines.append(f"  Media = {arr.mean():.4f}, Mediana = {np.median(arr):.4f}, DesvStd = {arr.std(ddof=0):.4f}\n")
        lines.append(f"  Min = {arr.min():.4f}, Max = {arr.max():.4f}\n")
        return "".join(lines)

    def _draw_app3(self, values, ref):
        self.app3_fig.clear()
        ax = self.app3_fig.add_subplot(111)
        x = list(range(1, len(values)+1))
        ax.plot(x, values, marker='o', linestyle='-')
        ax.axhline(ref, color='red', linestyle='--', label=f'Ref = {ref:.3f}')
        ax.set_title("Serie temporal del proceso")
        ax.set_xlabel("Obs")
        ax.set_ylabel("Valor")
        ax.legend()
        self.app3_canvas.draw()

    # -----------------------------
    # REPORTS / EXPORT
    # -----------------------------
    def _build_reports(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text="Análisis / Export")

        ttk.Label(frame, text="En esta pestaña puedes exportar los resultados de cada aplicación.").pack(padx=6, pady=6)
        btnf = ttk.Frame(frame); btnf.pack(padx=6, pady=6)
        ttk.Button(btnf, text="Exportar App1", command=lambda: self._export_text(self.app1_output)).pack(side='left', padx=6)
        ttk.Button(btnf, text="Exportar App2", command=lambda: self._export_text(self.app2_output)).pack(side='left', padx=6)
        ttk.Button(btnf, text="Exportar App3", command=lambda: self._export_text(self.app3_output)).pack(side='left', padx=6)

    def _export_text(self, text):
        if not text:
            messagebox.showinfo("Exportar", "No hay texto para exportar")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text","*.txt")])
        if not fn:
            return
        with open(fn, 'w', encoding='utf8') as f:
            f.write(text)
        messagebox.showinfo("Exportar", f"Exportado a {fn}")

    def _export_png(self, figure):
        fn = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image","*.png")]
        )
        if not fn:
            return
        try:
            figure.savefig(fn, dpi=300)
            messagebox.showinfo("Exportar PNG", f"Gráfico exportado a {fn}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = RunsApp(root)
    root.mainloop()