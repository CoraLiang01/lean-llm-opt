**Symbolic Model**

Let  

* \(I=\{1,\dots ,120\}\) – set of option indices (identified by the *Option* column).  
* \(J=\{1,\dots ,6\}\) – set of asset indices.  

Parameters (all taken from the supplied tables or given constants)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(c_i\) | per‑contract cost of option \(i\) | `file_0_view_0.Cost` |
| \(\Delta_i\) | per‑contract Delta of option \(i\) | `file_0_view_0.Delta` |
| \(\Gamma_i\) | per‑contract Gamma of option \(i\) | `file_0_view_0.Gamma` |
| \(\nu_i\) | per‑contract Vega of option \(i\) | `file_0_view_0.Vega` |
| \(L_i\) | maximum long position for option \(i\) | `file_0_view_0.MaxLong` |
| \(S_i\) | maximum short position for option \(i\) | `file_0_view_0.MaxShort` |
| \(a_{ij}\) | binary reference of option \(i\) to asset \(j\) | `file_1_view_0.Asset_k` (k = j) |
| \(\Delta^0\) | initial portfolio Delta | 0.25 |
| \(\Gamma^0\) | initial portfolio Gamma | 0.08 |
| \(\nu^0\) | initial portfolio Vega | 0.17 |
| \(\tau_\Delta\) | Delta tolerance | 0.06 |
| \(\tau_\Gamma\) | Gamma tolerance | 0.05 |
| \(\tau_\nu\) | Vega tolerance | 0.07 |

Decision variable  

* \(x_i \in \mathbb{Z}\) – number of contracts of option \(i\) (positive = long, negative = short).

---

### Objective
\[
\min \;\; \sum_{i\in I} c_i\,|x_i|
\]

### Constraints

1. **Delta band**

\[
\bigl|\;\Delta^0 + \sum_{i\in I}\sum_{j\in J}\Delta_i\,a_{ij}\,x_i\;\bigr|\;\le \;\tau_\Delta
\]

2. **Gamma band**

\[
\bigl|\;\Gamma^0 + \sum_{i\in I}\sum_{j\in J}\Gamma_i\,a_{ij}\,x_i\;\bigr|\;\le \;\tau_\Gamma
\]

3. **Vega band**

\[
\bigl|\;\nu^0 + \sum_{i\in I}\sum_{j\in J}\nu_i\,a_{ij}\,x_i\;\bigr|\;\le \;\tau_\nu
\]

4. **Trading limits**

\[
S_i \;\le\; x_i \;\le\; L_i , \qquad \forall i\in I
\]

---

**Data Mapping (CSVQA_DATA)**  

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(c_i\) | `file_0_view_0` | `Cost` |
| \(\Delta_i\) | `file_0_view_0` | `Delta` |
| \(\Gamma_i\) | `file_0_view_0` | `Gamma` |
| \(\nu_i\) | `file_0_view_0` | `Vega` |
| \(L_i\) | `file_0_view_0` | `MaxLong` |
| \(S_i\) | `file_0_view_0` | `MaxShort` |
| \(a_{ij}\) | `file_1_view_0` | `Asset_1` … `Asset_6` (column `Asset_k` for asset \(k\)) |
| \(x_i\) | – | – (decision variable) |

All constants (\(\Delta^0,\Gamma^0,\nu^0,\tau_\Delta,\tau_\Gamma,\tau_\nu\)) are fixed values supplied in the problem statement and not mapped to a table.