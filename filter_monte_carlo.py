from dataclasses import dataclass
import itertools
from typing import Tuple
import numpy as np


@dataclass
class FilterDesign:
    l_nh: float
    c_small_pf: float
    c_large_pf: float
    n_io_par: int
    n_mid_ser: int
    n_to_par: int
    n_tm_par: int


def calc_inductor_y(f_hz: np.ndarray, l_nh: np.ndarray, q_150: float = 85.0,
                    srf_hz: float = 1.9e9, l_via_nh: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    """Compute vectorized inductor admittance and esr across frequency and trials."""
    w = 2.0 * np.pi * f_hz
    l_h = l_nh * 1e-9
    r_esr = (w * l_h) / (q_150 * np.sqrt(f_hz / 150.0e6))
    c_par = 1.0 / (((2.0 * np.pi * srf_hz) ** 2) * l_h)

    y_core = (1.0 / (r_esr + 1j * w * l_h)) + (1j * w * c_par)
    z_tot = (1.0 / y_core) + (1j * w * l_via_nh * 1e-9)
    return (1.0 / z_tot), r_esr


def calc_cap_y(f_hz: np.ndarray, c_pf: np.ndarray, n_par: int, n_ser: int = 1,
               l_via_nh: float = 0.0, q_150: float = 1000.0, esl_nh: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Compute vectorized capacitor bank admittance and effective esr."""
    w = 2.0 * np.pi * f_hz
    c_f = c_pf * 1e-12
    r_esr = (1.0 / (w * c_f * q_150)) * (n_ser / n_par)
    x_single = (w * esl_nh * 1e-9) - (1.0 / (w * c_f))

    z_tot = r_esr + 1j * (x_single * (n_ser / n_par) + w * l_via_nh * 1e-9)
    return (1.0 / z_tot), r_esr


def solve_network_batch(f_hz: np.ndarray, des: FilterDesign, tol: np.ndarray,
                        p_avail_w: float = 5.0) -> dict:
    """Solve 3-node admittance tensor across all frequencies and trials simultaneously."""
    z0 = 50.0
    w = 2.0 * np.pi * f_hz

    y_cin, r_cin = calc_cap_y(f_hz, des.c_small_pf * tol[..., 3], des.n_io_par)
    y_cout, r_cout = calc_cap_y(f_hz, des.c_small_pf * tol[..., 4], des.n_io_par)
    y_c12, _ = calc_cap_y(f_hz, des.c_small_pf * tol[..., 5], 1, des.n_mid_ser)
    y_c23, _ = calc_cap_y(f_hz, des.c_small_pf * tol[..., 6], 1, des.n_mid_ser)

    z_in = z0 + (1.0 / y_cin)
    z_out = z0 + (1.0 / y_cout)

    y_l1, r_l1 = calc_inductor_y(f_hz, des.l_nh * tol[..., 0])
    y_l2, r_l2 = calc_inductor_y(f_hz, des.l_nh * tol[..., 1])
    y_l3, r_l3 = calc_inductor_y(f_hz, des.l_nh * tol[..., 2])

    y_ct1, r_ct1 = calc_cap_y(f_hz, des.c_large_pf * tol[..., 7], des.n_to_par, l_via_nh=0.35)
    y_ct2, r_ct2 = calc_cap_y(f_hz, des.c_large_pf * tol[..., 8], des.n_tm_par, l_via_nh=0.35)
    y_ct3, r_ct3 = calc_cap_y(f_hz, des.c_large_pf * tol[..., 9], des.n_to_par, l_via_nh=0.35)

    y_pcb = 1j * w * (0.65e-12 * tol[..., 10])
    y_sh1, y_sh2, y_sh3 = y_l1 + y_ct1 + y_pcb, y_l2 + y_ct2 + y_pcb, y_l3 + y_ct3 + y_pcb

    # stack 3x3 admittance tensor along trailing dimensions
    zeros = np.zeros_like(y_sh1)
    y_mat = np.stack([
        np.stack([(1.0 / z_in) + y_sh1 + y_c12, -y_c12, zeros], axis=-1),
        np.stack([-y_c12, y_c12 + y_sh2 + y_c23, -y_c23], axis=-1),
        np.stack([zeros, -y_c23, y_c23 + y_sh3 + (1.0 / z_out)], axis=-1)
    ], axis=-2)

    v_src = np.sqrt(4.0 * z0 * p_avail_w)
    i_rhs = np.stack([v_src / z_in, zeros, zeros], axis=-1)[..., np.newaxis]

    # solve all trials and frequencies in one lapack call
    v_nodes = np.linalg.solve(y_mat, i_rhs)[..., 0]

    i_in = (v_src - v_nodes[..., 0]) / z_in
    z_port1 = (v_src - i_in * z0) / i_in
    s11_mag = np.abs((z_port1 - z0) / (z_port1 + z0))
    s21_mag = np.abs(2.0 * (v_nodes[..., 2] / z_out) * z0 / v_src)

    i_l = np.stack([np.abs(v_nodes[..., 0] * y_l1),
                    np.abs(v_nodes[..., 1] * y_l2),
                    np.abs(v_nodes[..., 2] * y_l3)], axis=-1)
    p_l = (i_l ** 2) * np.stack([r_l1, r_l2, r_l3], axis=-1)

    p_caps = ((np.abs(i_in) ** 2) * r_cin +
              (np.abs(v_nodes[..., 2] / z_out) ** 2) * r_cout +
              (np.abs(v_nodes[..., 0] * y_ct1) ** 2) * r_ct1 +
              (np.abs(v_nodes[..., 1] * y_ct2) ** 2) * r_ct2 +
              (np.abs(v_nodes[..., 2] * y_ct3) ** 2) * r_ct3)

    return {
        "il_db": -20.0 * np.log10(np.clip(s21_mag, 1e-12, None)),
        "vswr": (1.0 + s11_mag) / np.clip(1.0 - s11_mag, 1e-6, None),
        "i_ind_rms": i_l,
        "p_ind_w": p_l,
        "p_tot_w": np.sum(p_l, axis=-1) + p_caps,
        "v_pep_pk": 2.0 * np.abs(v_nodes)
    }


def optimize_filter() -> Tuple[FilterDesign, dict, float]:
    """Vectorized search over discrete e24 grid with batch monte carlo yield."""
    f_eval = np.array([80e6, 98e6, 110e6, 144e6, 145e6, 146e6, 288e6, 432e6, 576e6, 720e6, 1022e6])
    idx_fm, idx_pass, idx_harm = slice(0, 3), slice(3, 6), slice(6, 11)

    f_mc = np.array([110e6, 144e6, 145e6, 146e6, 288e6])[np.newaxis, :]
    rng = np.random.default_rng(42)
    z_mc = np.clip(rng.standard_normal((2000, 1, 11)), -3.0, 3.0)

    e24_small = [1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.2, 6.8, 7.5, 8.2, 10.0, 12.0]
    e24_large = [8.2, 9.1, 10.0, 11.0, 12.0, 13.0, 15.0, 18.0, 22.0, 24.0, 27.0, 30.0, 33.0]

    best_des, best_res, best_yield, best_score = None, None, 0.0, -1e9
    nom_tol = np.ones((len(f_eval), 11))

    # iterate over flat cartesian product of candidate parameters
    grid = itertools.product([27.0, 33.0], e24_small, e24_large, [1, 2], [1, 2], [1, 2, 3], [1, 2, 3, 4])
    for l_nh, c_sm, c_lg, n_io, n_mid, n_to, n_tm in grid:
        des = FilterDesign(l_nh, c_sm, c_lg, n_io, n_mid, n_to, n_tm)
        res = solve_network_batch(f_eval[:, np.newaxis], des, nom_tol)

        max_vswr = float(np.max(res["vswr"][idx_pass]))
        max_il = float(np.max(res["il_db"][idx_pass]))
        min_fm = float(np.min(res["il_db"][idx_fm]))
        min_hm = float(np.min(res["il_db"][idx_harm]))

        if max_vswr > 1.45 or max_il > 1.2 or min_fm < 20.0 or min_hm < 30.0:
            continue

        # compute 2000 monte carlo trials x 5 frequencies in one vectorized call
        sig = np.array([
            0.02 / 3.0, 0.02 / 3.0, 0.02 / 3.0,
            (0.01 / 3.0) / np.sqrt(n_io), (0.01 / 3.0) / np.sqrt(n_io),
            (0.01 / 3.0) / np.sqrt(n_mid), (0.01 / 3.0) / np.sqrt(n_mid),
            (0.01 / 3.0) / np.sqrt(n_to), (0.01 / 3.0) / np.sqrt(n_tm), (0.01 / 3.0) / np.sqrt(n_to),
            0.08
        ])[np.newaxis, np.newaxis, :]

        mc = solve_network_batch(f_mc, des, 1.0 + z_mc * sig)
        passed = ((np.max(mc["vswr"][:, 1:4], axis=1) <= 2.0) &
                  (np.max(mc["il_db"][:, 1:4], axis=1) <= 2.0) &
                  (mc["il_db"][:, 0] >= 15.0) &
                  (mc["il_db"][:, 4] >= 25.0))
        mc_yield = 100.0 * float(np.mean(passed))

        if mc_yield >= 90.0:
            score = mc_yield * 2.0 - max_il * 25.0 - (max_vswr - 1.0) * 30.0 + min(min_fm, 40.0) * 0.5
            if score > best_score:
                best_score, best_des, best_res, best_yield = score, des, res, mc_yield

    return best_des, best_res, best_yield


if __name__ == "__main__":
    des, res, mc_yield = optimize_filter()
    print(f"Inductor: {des.l_nh} nH | Caps: {des.c_small_pf} pF, {des.c_large_pf} pF")
    print(f"Banks   : C_io={des.n_io_par}p, C_mid={des.n_mid_ser}s, C_to={des.n_to_par}p, C_tm={des.n_tm_par}p")
    print(f"Yield   : {mc_yield:.1f}% | Max IL: {np.max(res['il_db'][3:6]):.2f} dB | Max VSWR: {np.max(res['vswr'][3:6]):.2f}:1")
    print(f"Stopband: FM >= {np.min(res['il_db'][0:3]):.1f} dB | Harmonics >= {np.min(res['il_db'][6:11]):.1f} dB")
    print(f"5W Heat : {np.max(res['p_tot_w'][3:6]):.2f} W | L2 Max I_rms: {np.max(res['i_ind_rms'][3:6, 0, 1]):.2f} A")