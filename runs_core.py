# runs_core.py
"""
Módulo core para pruebas de corridas (runs test).
Funciones:
 - combos(n,k): combinatoria segura
 - exact_runs_distribution(n1, n2): distribuciones exactas P(R)
 - exact_prob_R(n1, n2, R): P(R = R) (usa exact_runs_distribution)
 - runs_count_binary(seq): cuenta corridas en secuencia binaria (0/1)
 - runs_test_binary(seq): estadístico Z y p-value (aprox. normal)
 - runs_test_numeric(values, reference): transforma a arriba/abajo y llama a runs_test_binary
 - summarize_runs_lengths(seq): devuelve lista de corridas (valor,longitud)
"""
import math
from typing import List, Tuple, Dict
import numpy as np
from scipy.stats import norm

def combos(n: int, k: int) -> int:
    """Combinación n choose k con validación."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)

def exact_runs_distribution(n1: int, n2: int) -> Dict[int, float]:
    """
    Calcula la distribución exacta de P(R) para dos tipos (n1 de tipo A, n2 de tipo B).
    Devuelve diccionario {R: P(R)}.
    Fórmulas estándar:
      - R par (R=2k): ways = 2 * C(n1-1, k-1) * C(n2-1, k-1)
      - R impar (R=2k+1): ways = C(n1-1,k)*C(n2-1,k-1) + C(n1-1,k-1)*C(n2-1,k)
    Referencia: combinatoria clásica de corridas.
    """
    n = n1 + n2
    total = combos(n, n1)
    probs = {}
    if total == 0:
        return probs
    for R in range(1, n+1):
        if R % 2 == 0:
            k = R // 2
            ways = 2 * combos(n1-1, k-1) * combos(n2-1, k-1)
        else:
            k = R // 2
            ways = combos(n1-1, k) * combos(n2-1, k-1) + combos(n1-1, k-1) * combos(n2-1, k)
        if ways > 0:
            probs[R] = ways / total
    return probs

def exact_prob_R(n1: int, n2: int, R: int) -> float:
    """P(R = R) usando distribución exacta."""
    dist = exact_runs_distribution(n1, n2)
    return dist.get(R, 0.0)

def runs_count_binary(seq: List[int]) -> int:
    """Cuenta corridas en secuencia binaria (0/1)."""
    if not seq:
        return 0
    runs = 1
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            runs += 1
    return runs

def runs_test_binary(seq: List[int]) -> dict:
    """
    Runs test para secuencia binaria.
    Devuelve un dict con: n, n1, n2, runs, mu, var, sigma, Z, p_value.
    Usa aproximación normal: mu = 2*n1*n2/n + 1
    var = (2*n1*n2*(2*n1*n2 - n)) / (n^2 * (n-1))
    p-value bilateral ~ 2*(1 - Phi(|Z|))
    Nota: la aproximación requiere n1>0, n2>0 y n>1.
    """
    n = len(seq)
    n1 = sum(seq)
    n2 = n - n1
    runs = runs_count_binary(seq)
    result = {'n': n, 'n1': n1, 'n2': n2, 'runs': runs}
    if n1 > 0 and n2 > 0 and n > 1:
        mu = (2.0 * n1 * n2) / n + 1.0
        var = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n**2 * (n - 1))
        sigma = math.sqrt(var) if var > 0 else 0.0
        Z = (runs - mu) / sigma if sigma > 0 else 0.0
        p_value = 2.0 * (1.0 - norm.cdf(abs(Z))) if sigma > 0 else 1.0
        result.update({'mu': mu, 'var': var, 'sigma': sigma, 'Z': Z, 'p_value': p_value})
    else:
        # Casos degenerados (todo 0 o todo 1)
        result.update({'mu': None, 'var': None, 'sigma': None, 'Z': None, 'p_value': None})
    return result

def summarize_runs_lengths(seq: List[int]) -> List[Tuple[int,int]]:
    """Devuelve lista de (valor, longitud) de cada corrida en la secuencia."""
    if not seq:
        return []
    runs = []
    current = seq[0]
    length = 1
    for x in seq[1:]:
        if x == current:
            length += 1
        else:
            runs.append((current, length))
            current = x
            length = 1
    runs.append((current, length))
    return runs

def runs_test_numeric(values: List[float], reference: float) -> dict:
    """
    Aplica runs test a una serie de valores numéricos comparándolos con 'reference'.
    Transforma cada valor a 1 si v > reference, 0 si v <= reference (puedes cambiar la regla si lo deseas).
    Devuelve: dict con la prueba binaria y estadísticas descriptivas.
    """
    signs = [1 if v > reference else 0 for v in values]
    stats = runs_test_binary(signs)
    arr = np.array(values)
    descr = {
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std(ddof=0)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'n': len(values)
    }
    return {'signs': signs, 'binary_stats': stats, 'descr': descr}
