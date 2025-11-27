# klein_resonance_map_fixed.py
# Cleaned and runnable version of the Klein-map pipeline

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------
# 1) Site data (lat, lon in degrees)
# -------------------------
sites = [
    ("Bermuda Triangle",    25.5,   -69.0),
    ("Michigan Triangle",   43.4,   -86.9),
    ("Giza Pyramid",        29.98,   31.14),
    ("Chichen Itza",        20.68,  -88.57),
    ("Teotihuacan",         19.69,  -98.84),
    ("Stonehenge",          51.18,   -1.83),
    ("Avebury",             51.43,   -1.85),
    ("Newgrange",           53.69,   -6.48),
    ("Arbor Low",           53.08,   -1.76),
    ("Rollright Stones",    51.97,   -1.57),
    ("Hal Saflieni Hypogeum",35.83,   14.51),
]

df = pd.DataFrame(sites, columns=["Site", "lat_deg", "lon_deg"])
df["lat_rad"] = np.deg2rad(df["lat_deg"])
df["lon_rad"] = np.deg2rad(df["lon_deg"])


# -------------------------
# 2) Geometry / algebra helpers
# -------------------------
def antipode(lat_deg, lon_deg):
    """
    Antipodal point: latitude -> -latitude, longitude -> lon + 180 normalized to [-180,180)
    """
    lat_a = -lat_deg
    lon_a = (lon_deg + 180.0 + 180.0) % 360.0 - 180.0  # normalize to [-180, 180)
    return lat_a, lon_a


def sph_to_cart(lat_deg, lon_deg):
    """
    Convert geographic latitude (deg) and longitude (deg) to unit-sphere Cartesian coords.
      lat  ∈ [-90,90] : latitude (φ)
      lon  ∈ [-180,180)
    Uses:
      x = cos(lat) * cos(lon)
      y = cos(lat) * sin(lon)
      z = sin(lat)
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z])


def stereographic_from_sphere(cart):
    """
    Stereographic projection from unit sphere (x,y,z) to complex plane:
      if point is (x,y,z) on unit sphere, map to
        u = x / (1 - z), v = y / (1 - z)  (except when z ~ 1)
      complex = u + i v
    This is the projection from north pole (0,0,1) to plane z=0.
    """
    x, y, z = cart
    denom = 1.0 - z
    eps = 1e-12
    if denom < eps:
        # point near north pole → return a very large complex value
        denom = eps
    u = x / denom
    v = y / denom
    return u + 1j * v


def quaternion_pair_from_complex(c_p, c_a):
    """
    Form a 4-vector quaternion-like object [w, i, j, k]
    from two complex numbers: Cp and Ca
    """
    return np.array([c_p.real, c_p.imag, c_a.real, c_a.imag], dtype=float)


def cayley_like_klein_map(q):
    """
    Simple Cayley-like map: q -> q / (1 + ||q||^2)
    Operates elementwise on 4-vector q.
    """
    norm2 = np.sum(q**2)
    if norm2 == 0.0:
        return q.copy()
    return q / (1.0 + norm2)


def quaternion_conjugate(q):
    """
    Conjugate of quaternion-like vector [w, i, j, k] -> [w, -i, -j, -k]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def curvature_dot(Kp, Ka):
    """
    Real scalar curvature proxy: dot(Kp, conj(Ka))
    """
    conj_Ka = quaternion_conjugate(Ka)
    return float(np.dot(Kp, conj_Ka))


# -------------------------
# 3) Apply map to all sites
# -------------------------
rows = []
proj_points = []  # (site, x, y, z) for plotting

for _, row in df.iterrows():
    site = row["Site"]
    lat = float(row["lat_deg"])
    lon = float(row["lon_deg"])

    # stereographic complex for P
    cart_p = sph_to_cart(lat, lon)
    Cp = stereographic_from_sphere(cart_p)

    # antipode -> stereographic complex for A(P)
    lat_a, lon_a = antipode(lat, lon)
    cart_a = sph_to_cart(lat_a, lon_a)
    Ca = stereographic_from_sphere(cart_a)

    # quaternion-like Q(P)
    q = quaternion_pair_from_complex(Cp, Ca)

    # normalize Q to unit length for numerical stability (optional)
    qnorm = np.linalg.norm(q)
    if qnorm > 0:
        q_unit = q / qnorm
    else:
        q_unit = q.copy()

    # Klein-like map
    Kp = cayley_like_klein_map(q_unit)

    # For Ka compute from antipodal quaternion (do not assume Ka == -Kp)
    q_a = quaternion_pair_from_complex(Ca, Cp)  # swap order for antipode quaternion
    q_a_norm = np.linalg.norm(q_a)
    q_a_unit = q_a / q_a_norm if q_a_norm > 0 else q_a.copy()
    Ka = cayley_like_klein_map(q_a_unit)

    # curvature proxy
    kappa = curvature_dot(Kp, Ka)

    # project to 3D for plotting: use the vector (i, j, k) components (drop scalar part w)
    x, y, z = Kp[1], Kp[2], Kp[3]

    rows.append({
        "Site": site,
        "lat_deg": lat,
        "lon_deg": lon,
        "Cp_real": Cp.real,
        "Cp_imag": Cp.imag,
        "Ca_real": Ca.real,
        "Ca_imag": Ca.imag,
        "q_w": float(q_unit[0]),
        "q_i": float(q_unit[1]),
        "q_j": float(q_unit[2]),
        "q_k": float(q_unit[3]),
        "K_w": float(Kp[0]),
        "K_i": float(Kp[1]),
        "K_j": float(Kp[2]),
        "K_k": float(Kp[3]),
        "kappa": kappa,
        "x": x, "y": y, "z": z
    })

    proj_points.append((site, x, y, z))

df_out = pd.DataFrame(rows)

# -------------------------
# 4) Print numeric table & curvature stats
# -------------------------
print("\n=== KLEIN MAP NUMERIC TABLE ===")
print(df_out[["Site", "lat_deg", "lon_deg", "K_w", "K_i", "K_j", "K_k", "kappa"]].to_string(index=False))

kappa_abs = np.abs(df_out["kappa"].values)
print("\nCurvature stats (kappa): min, mean, max =",
      float(kappa_abs.min()), float(kappa_abs.mean()), float(kappa_abs.max()))

# -------------------------
# 5) Plotly 3D visualization: quaternion-vector components as points + parametric Klein bottle surface
# -------------------------
fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]],
                    subplot_titles=["Klein-like Map: quaternion vector components (i,j,k)"])

# scatter site points
marker_colors = []
for name, x, y, z in proj_points:
    if "Triangle" in name:
        marker_colors.append("red")
    elif "Pyramid" in name or "Hypogeum" in name:
        marker_colors.append("gold")
    else:
        marker_colors.append("cyan")

scatter = go.Scatter3d(
    x=[p[1] for p in proj_points],
    y=[p[2] for p in proj_points],
    z=[p[3] for p in proj_points],
    mode="markers+text",
    marker=dict(size=6, color=marker_colors),
    text=[p[0] for p in proj_points],
    textposition="top center",
    name="Sites"
)
fig.add_trace(scatter)

# parametric Klein bottle surface (standard parametric form)
# Source: common parametric Klein bottle used for visualization
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, 2 * np.pi, 100)
U, V = np.meshgrid(u, v)

# Parametric Klein bottle (immersed in R^3)
# using a common representation (not unique)
# see e.g. https://en.wikipedia.org/wiki/Klein_bottle#Parametrization
r = 4.0
X = (r + np.cos(U / 2.0) * np.sin(V) - np.sin(U / 2.0) * np.sin(2 * V)) * np.cos(U)
Y = (r + np.cos(U / 2.0) * np.sin(V) - np.sin(U / 2.0) * np.sin(2 * V)) * np.sin(U)
Z = np.sin(U / 2.0) * np.sin(V) + np.cos(U / 2.0) * np.sin(2 * V)

fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.35, colorscale="Viridis", showscale=False, name="Klein surface"))

fig.update_layout(
    scene=dict(
        xaxis_title="i (quat)",
        yaxis_title="j (quat)",
        zaxis_title="k (quat)",
        aspectmode="cube"
    ),
    title="Klein-like Quaternion Map: sites projected to quaternion vector components (i,j,k)",
    height=800
)

# show interactive plot (in notebook or browser)
fig.show()

# -------------------------
# 6) Save CSV
# -------------------------
df_out.to_csv("klein_resonance_map_fixed.csv", index=False)
print("\nSaved numeric results to 'klein_resonance_map_fixed.csv'")
