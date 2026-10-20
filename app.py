import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y TEMA
# ==========================================
st.set_page_config(
    page_title="Apex Optimization - Electrostatic Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Apex Optimization Technologies")
st.caption("Simulador Interactivo de Equilibrio Electrostático y Optimización Numérica")
st.markdown("---")

# Constantes Físicas
K_E = 8.9875e9      # Constante de Coulomb (N·m²/C²)
EPSILON_REG = 1e-4  # Factor de suavizado para evitar división por cero

# ==========================================
# BARRA LATERAL (CONTROLES Y PARÁMETROS)
# ==========================================
st.sidebar.header("⚙️ Parámetros del Algoritmo")

scenario = st.sidebar.selectbox(
    "Escenario de Prueba",
    ["1D: Validación Analítica (2 Cargas)", "2D: Configuración Multicarga (3 Cargas)"]
)

eta = st.sidebar.select_slider(
    "Tasa de Aprendizaje (η)",
    options=[1e-13, 1e-12, 5e-12, 1e-11, 5e-11, 1e-10],
    value=1e-11,
    format_func=lambda x: f"{x:.1e}"
)

max_iter = st.sidebar.slider(
    "Límite de Iteraciones", 
    min_value=50, 
    max_value=2000, 
    value=300, 
    step=50
)

tol_E = st.sidebar.select_slider(
    "Tolerancia Residual δ_E (N/C)",
    options=[1e-2, 1e-3, 1e-4, 1e-6],
    value=1e-3,
    format_func=lambda x: f"{x:.1e}"
)

st.sidebar.markdown("---")
st.sidebar.header("📍 Condición Inicial (Semilla)")

if "1D" in scenario:
    x0 = st.sidebar.slider("Posición Inicial x₀ (m)", min_value=0.1, max_value=3.9, value=0.5, step=0.1)
    y0 = 0.0
else:
    x0 = st.sidebar.slider("Posición Inicial x₀ (m)", min_value=-3.0, max_value=3.0, value=0.5, step=0.1)
    y0 = st.sidebar.slider("Posición Inicial y₀ (m)", min_value=-3.0, max_value=3.0, value=1.5, step=0.1)

r_init = np.array([x0, y0])

# ==========================================
# FUNCIONES FÍSICO-MATEMÁTICAS
# ==========================================
def electric_field(r, charges):
    """Calcula el vector Campo Eléctrico E(r) en N/C."""
    E = np.zeros(2)
    for c in charges:
        diff = r - c['pos']
        dist_sq = np.sum(diff**2) + EPSILON_REG
        dist = np.sqrt(dist_sq)
        E += K_E * c['q'] * diff / (dist**3)
    return E

def cost_function(r, charges):
    """Función de costo escalar f(r) = |E(r)|^2."""
    E = electric_field(r, charges)
    return np.sum(E**2)

def cost_gradient(r, charges, h=1e-5):
    """Cálculo del gradiente de f(r) por diferencias finitas centrales."""
    grad = np.zeros(2)
    for i in range(2):
        r_p, r_m = r.copy(), r.copy()
        r_p[i] += h
        r_m[i] -= h
        grad[i] = (cost_function(r_p, charges) - cost_function(r_m, charges)) / (2 * h)
    return grad

def run_gradient_descent(r_0, eta_val, max_i, tol, charges):
    """Ejecuta la optimización iterativa por Descenso del Gradiente."""
    trajectory = []
    costs = []
    e_residuals = []
    
    r = r_0.astype(float)
    for k in range(max_i):
        E_vec = electric_field(r, charges)
        E_norm = np.linalg.norm(E_vec)
        cost = np.sum(E_vec**2)
        
        trajectory.append(r.copy())
        costs.append(cost)
        e_residuals.append(E_norm)
        
        if E_norm < tol:
            break
            
        grad = cost_gradient(r, charges)
        r -= eta_val * grad
        
    return np.array(trajectory), np.array(costs), np.array(e_residuals)

# ==========================================
# CONFIGURACIÓN DE ESCENARIOS
# ==========================================
if "1D" in scenario:
    st.subheader("📌 Escenario 1D: Validación Analítica con Dos Cargas en el Eje X")
    
    col1, col2 = st.columns(2)
    with col1:
        q1_uC = st.slider("Carga 1 (q₁) en x = 0 m (μC)", 1.0, 10.0, 4.0, 0.5)
    with col2:
        q2_uC = st.slider("Carga 2 (q₂) en x = 4 m (μC)", 1.0, 10.0, 1.0, 0.5)
        
    L = 4.0
    charges = [
        {'pos': np.array([0.0, 0.0]), 'q': q1_uC * 1e-6},
        {'pos': np.array([L, 0.0]),   'q': q2_uC * 1e-6}
    ]
    
    # Solución Analítica teórica
    q1, q2 = q1_uC * 1e-6, q2_uC * 1e-6
    x_analitico = (L * np.sqrt(q1)) / (np.sqrt(q1) + np.sqrt(q2))
    r_analitico = np.array([x_analitico, 0.0])

else:
    st.subheader("📌 Escenario 2D: Configuración Multicarga en el Plano XY")
    charges = [
        {'pos': np.array([-2.0, 0.0]), 'q': 2e-6},
        {'pos': np.array([2.0, 0.0]),  'q': 2e-6},
        {'pos': np.array([0.0, 2.0]),  'q': -1e-6}
    ]
    r_analitico = None

# Ejecutar Algoritmo
trajectory, costs, e_residuals = run_gradient_descent(r_init, eta, max_iter, tol_E, charges)
r_final = trajectory[-1]
E_final = e_residuals[-1]
num_steps = len(costs)

# ==========================================
# DESPLIEGUE DE MÉTRICAS
# ==========================================
st.markdown("### 📊 Métricas de Convergencia")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Posición Final (x, y)", f"({r_final[0]:.4f}, {r_final[1]:.4f}) m")
m2.metric("Residual Campo |E|", f"{E_final:.2e} N/C")
m3.metric("Iteraciones", f"{num_steps} / {max_iter}")

if r_analitico is not None:
    delta_r = np.linalg.norm(r_final - r_analitico)
    m4.metric("Error δ_r (vs Analítico)", f"{delta_r:.4e} m")
else:
    m4.metric("Solución Analítica", "Multicarga (2D)")

st.markdown("---")

# ==========================================
# VISUALIZACIÓN GRÁFICA
# ==========================================
col_plot1, col_plot2, col_table = st.columns([1.2, 1, 0.8])

# 1. Gráfico Principal (Mapa / Trayectoria)
fig_map, ax_map = plt.subplots(figsize=(6, 5))

if "1D" in scenario:
    x_vals = np.linspace(-0.5, 4.5, 200)
    f_vals = [cost_function(np.array([x, 0.0]), charges) for x in x_vals]
    
    ax_map.plot(x_vals, f_vals, 'k--', alpha=0.4, label=r"Costo $f(x) = |E|^2$")
    ax_map.axvline(x_analitico, color='r', linestyle=':', linewidth=2, label=f"Analítico ({x_analitico:.3f} m)")
    ax_map.plot(trajectory[:, 0], costs, 'g.-', label="Iteraciones")
    ax_map.plot(r_final[0], cost_function(r_final, charges), 'ro', markersize=8, label="Equilibrio")
    ax_map.set_xlabel("Posición X (m)")
    ax_map.set_ylabel("Costo f(x)")
    ax_map.set_yscale("log")
    ax_map.set_title("Trayectoria sobre Perfil de Costo 1D")
    ax_map.legend()
    ax_map.grid(True)

else:
    grid_x, grid_y = np.meshgrid(np.linspace(-3.5, 3.5, 25), np.linspace(-1.5, 3.5, 25))
    Ex, Ey = np.zeros_like(grid_x), np.zeros_like(grid_y)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            E_v = electric_field(np.array([grid_x[i, j], grid_y[i, j]]), charges)
            Ex[i, j], Ey[i, j] = E_v[0], E_v[1]
            
    EMag = np.sqrt(Ex**2 + Ey**2)
    ax_map.quiver(grid_x, grid_y, Ex/(EMag+1e-9), Ey/(EMag+1e-9), np.log10(EMag+1e-9), cmap='plasma', alpha=0.5)
    
    for c in charges:
        color = 'red' if c['q'] > 0 else 'blue'
        ax_map.plot(c['pos'][0], c['pos'][1], 'o', color=color, markersize=10)
        
    ax_map.plot(trajectory[:, 0], trajectory[:, 1], 'g.-', label="Trayectoria Algoritmo")
    ax_map.plot(r_init[0], r_init[1], 'go', label="Semilla r⁽⁰⁾")
    ax_map.plot(r_final[0], r_final[1], 'r*', markersize=12, label="Equilibrio r_eq")
    ax_map.set_title("Campo Vectorial E(x,y) y Trayectoria")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.legend()
    ax_map.grid(True)

with col_plot1:
    st.pyplot(fig_map)

# 2. Gráfico de Convergencia
fig_conv, ax_conv = plt.subplots(figsize=(5, 4.5))
ax_conv.plot(e_residuals, 'b-', linewidth=2)
ax_conv.axhline(tol_E, color='r', linestyle='--', label="Tolerancia δ_E")
ax_conv.set_yscale("log")
ax_conv.set_title("Evolución del Residual de Campo |E|")
ax_conv.set_xlabel("Iteración (k)")
ax_conv.set_ylabel("|E| (N/C)")
ax_conv.legend()
ax_conv.grid(True)

with col_plot2:
    st.pyplot(fig_conv)

# 3. Tabla de Datos
with col_table:
    st.markdown("**Tabla de Iteraciones**")
    df_history = pd.DataFrame({
        "Iteración": np.arange(num_steps),
        "X (m)": trajectory[:, 0],
        "Y (m)": trajectory[:, 1],
        "|E| (N/C)": e_residuals
    })
    st.dataframe(
        df_history.style.format({"X (m)": "{:.4f}", "Y (m)": "{:.4f}", "|E| (N/C)": "{:.2e}"}), 
        height=380
    )