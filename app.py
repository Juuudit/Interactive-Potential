import os
import sys

# Get absolute path to the 'Vs' folder and add it to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ============================================================
# Numerical Safety Helpers & Constants
# ============================================================

LOG_EPS = 1e-300
ab = 5.4076
af = 2.6351


def log_real(z):
    zc = np.asarray(z, dtype=complex)
    zc = np.where(np.abs(zc) < LOG_EPS, LOG_EPS + 0j, zc)
    return np.real(np.log(zc))


def pow32_real(x):
    return np.real(np.asarray(x, dtype=complex) ** 1.5)


# ============================================================
# Running of Couplings and Field (RGEs)
# ============================================================


def beta_functions(mu, couplings):
    """
    couplings = [gsq, λS, y, mSq, G_field]
    """
    gsq, lamS, y, mSq, G_field = couplings
    gamma_field = (3 * gsq - y**2) / (16 * np.pi**2)
    beta_gsq = gsq**2 / (12 * np.pi**2)
    beta_lamS = (1 / (8 * np.pi**2)) * (
        3 * gsq**2 - y**4 - 6 * gsq * lamS + 2 * lamS * (y**2 + 5 * lamS)
    )
    beta_y = (1 / (32 * np.pi**2)) * (-3 * gsq * y + 4 * y**3)
    beta_mSq = (1 / (8 * np.pi**2)) * mSq * (-3 * gsq + y**2 + 4 * lamS)
    beta_G = -gamma_field * G_field

    return np.array([beta_gsq, beta_lamS, beta_y, beta_mSq, beta_G])


def solve_rges(couplings_init, mu_init, mu_range, n_points=1000):
    mu_min, mu_max = mu_range

    def rhs(mu, c):
        return beta_functions(mu, c) / mu

    n_low = n_points // 2
    n_high = n_points - n_low

    mu_low = np.logspace(
        np.log10(mu_min), np.log10(mu_init), num=n_low, endpoint=False
    )
    mu_high = np.logspace(np.log10(mu_init), np.log10(mu_max), num=n_high)

    sol_fwd = solve_ivp(
        rhs,
        (mu_init, mu_max),
        couplings_init,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )

    sol_bwd = solve_ivp(
        rhs,
        (mu_init, mu_min),
        couplings_init,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )

    C_low = sol_bwd.sol(mu_low)
    C_high = sol_fwd.sol(mu_high)

    mu_vals = np.concatenate([mu_low, mu_high])
    couplings_vals = np.hstack([C_low, C_high])

    return mu_vals, couplings_vals


# ============================================================
# Field-dependent masses & Thermal masses
# ============================================================


def mphi2(phi, mSq, lamS):
    return -mSq + 3.0 * lamS * phi**2


def msigma2(phi, mSq, lamS):
    return -mSq + lamS * phi**2


def mAp2(phi, gsq):
    return gsq * phi**2


def mchi(phi, y):
    return y * phi / np.sqrt(2.0)


def Pi_phi(gsq, lamS, T, y):
    return (lamS * T**2) / 3.0 + (gsq * T**2) / 4.0 + (y**2 * T**2) / 12.0


def Pi_Ap(gsq, T):
    return (5.0 * gsq * T**2) / 12.0


# ============================================================
# Potentials (Tree, CW, HT, Daisy, Veff)
# ============================================================


def Vtree(phi, lamS, mSq):
    return np.real(-(mSq / 2.0) * phi**2 + (lamS / 4.0) * phi**4)


def VCW(phi, gsq, lamS, mSq, muR, y):
    mphi_sq = mphi2(phi, mSq, lamS)
    msigma_sq = msigma2(phi, mSq, lamS)
    mAp_sq = mAp2(phi, gsq)
    mchi_val = mchi(phi, y)

    term_phi = mphi_sq**2 * (log_real(mphi_sq / muR**2) - 1.5)
    term_sig = msigma_sq**2 * (log_real(msigma_sq / muR**2) - 1.5)
    term_Ap = 3.0 * mAp_sq**2 * (log_real(mAp_sq / muR**2) - 5.0 / 6.0)
    term_chi = -4.0 * mchi_val**4 * (log_real((mchi_val**2) / muR**2) - 1.5)

    return np.real(
        (1.0 / (64.0 * np.pi**2)) * (term_phi + term_sig + term_Ap + term_chi)
    )


def JBhigh(x):
    return np.real(
        -np.pi**4 / 45.0
        + (np.pi**2 / 12.0) * x
        - (np.pi / 6.0) * pow32_real(x)
        - (1.0 / 32.0) * x**2 * (log_real(x) - ab)
    )


def JFhigh(x):
    return np.real(
        7.0 * np.pi**4 / 360.0
        - (np.pi**2 / 24.0) * x
        - (1.0 / 32.0) * x**2 * (log_real(x) - af)
    )


def VThighT(phi, T, gsq, lamS, mSq, y):
    x_phi = mphi2(phi, mSq, lamS) / T**2
    x_sigma = msigma2(phi, mSq, lamS) / T**2
    x_Ap = mAp2(phi, gsq) / T**2
    x_chi = (mchi(phi, y) ** 2) / T**2

    return np.real(
        (T**4 / (2.0 * np.pi**2))
        * (
            JBhigh(x_phi)
            + JBhigh(x_sigma)
            + 3.0 * JBhigh(x_Ap)
            - 4.0 * JFhigh(x_chi)
        )
    )


def CorrTH(mi, Mi):
    return np.real(pow32_real(mi + Mi) - pow32_real(mi))


def Vdaisy(phi, gsq, lamS, mSq, T, y):
    PiPhi = Pi_phi(gsq, lamS, T, y)
    PiAp = Pi_Ap(gsq, T)

    return np.real(
        -(T / (12.0 * np.pi))
        * (
            CorrTH(mphi2(phi, mSq, lamS), PiPhi)
            + CorrTH(msigma2(phi, mSq, lamS), PiPhi)
            + CorrTH(mAp2(phi, gsq), PiAp)
        )
    )


def Veff4DHTRG(phi, T, gsq, lamS, mSq, muR, y):
    V = (
        lambda x: Vtree(x, lamS, mSq)
        + VCW(x, gsq, lamS, mSq, muR, y)
        + VThighT(x, T, gsq, lamS, mSq, y)
        + Vdaisy(x, gsq, lamS, mSq, T, y)
    )
    return np.real(V(phi) - V(0.0))


# ============================================================
# Main Potential Function Constructor
# ============================================================


def make_VphiRun(
    g0,
    y0,
    lam0=0.0,
    m0=0.0,
    mu_init=1,
    G0=1.0,
    mu_range=(1e-12, 1e11),
    n_points=1000,
    interp_kind="linear",
):
    couplings_init = np.array([g0**2, lam0, y0, m0, G0], dtype=float)

    mu_vals, C_vals = solve_rges(
        couplings_init=couplings_init,
        mu_init=mu_init,
        mu_range=mu_range,
        n_points=n_points,
    )

    interp_all = interp1d(
        np.log(mu_vals),
        C_vals,
        axis=1,
        kind=interp_kind,
        bounds_error=False,
        fill_value="extrapolate",
    )

    def get_couplings(muR):
        gsq, lam, y, mSq, G = interp_all(np.log(muR))
        gsq_val = float(gsq)
        return {
            "gsq": gsq_val,
            "g": np.sqrt(max(0.0, gsq_val)),
            "lam": float(lam),
            "y": float(y),
            "mSq": float(mSq),
            "G": float(G),
        }

    def VphiRun(phi, T, muR=None):
        if muR is None:
            muR = np.pi * T

        couplings = get_couplings(muR)
        phi_rescaled = couplings["G"] * phi

        return Veff4DHTRG(
            phi_rescaled,
            T,
            couplings["gsq"],
            couplings["lam"],
            couplings["mSq"],
            muR,
            couplings["y"],
        )

    VphiRun.get_couplings = get_couplings
    return VphiRun


# ============================================================
# Streamlit App Configuration & Layout
# ============================================================

st.set_page_config(page_title="Effective Potential Plot", layout="centered")

# Hide default Streamlit header/footer
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    </style>
""",
    unsafe_allow_html=True,
)

# Matplotlib Styling
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 16,
    }
)

st.title("Interactive Effective Potential")

# --- Interactive Sliders ---
col1, col2, col3 = st.columns(3)

with col1:
    T_val = st.slider(
        label=r"Temperature ($T$)",
        min_value=0.01,
        max_value=1.00,
        value=0.25,
        step=0.01,
    )

with col2:
    g0_var = st.slider(
        label=r"Gauge coupling ($g_0$)",
        min_value=0.00,
        max_value=1.00,
        value=0.80,
        step=0.05,
    )

with col3:
    y0_var = st.slider(
        label=r"Yukawa coupling ($y_0$)",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.05,
    )

# --- Calculation & Plotting ---
phi_vals = np.linspace(0, 2, 500)

V_func = make_VphiRun(g0=g0_var, y0=y0_var, lam0=0)
V_vals = np.array([V_func(phi, T=T_val) for phi in phi_vals])

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    phi_vals,
    V_vals,
    color="#3F889E",
    lw=4.5,
    label=rf"$T={T_val:.2f}\mu_0,\ g_0={g0_var:.2f},\ y_0={y0_var:.2f}$",
)

ax.axhline(y=0, color="grey", linestyle=":", linewidth=1.5, alpha=0.7)
ax.set_xlabel(r"$\phi$", fontsize=22, labelpad=8)
ax.set_ylabel(r"$V(\phi)$", fontsize=22, labelpad=8)  # Fixed typo here
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(phi_vals[0], phi_vals[-1])

# Dynamic y-axis scaling to accommodate shape changes gracefully
v_min, v_max = np.min(V_vals), np.max(V_vals)
margin = max(abs(v_min), abs(v_max), 0.001) * 0.25
ax.set_ylim(v_min - margin, v_max + margin)

ax.legend(
    loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=18
)

# Display plot in Streamlit
st.pyplot(fig)
