**Mathematical Model**

Let  

* \(I=\{1,\dots ,120\}\) index of options  
* \(J=\{1,\dots ,6\}\) index of assets  

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(c_i\) | per‑contract cost of option \(i\) | `OptionCharacteristics.csv` – column **Cost** |
| \(\Delta_i,\Gamma_i,\nu_i\) | per‑contract Delta, Gamma, Vega of option \(i\) | `OptionCharacteristics.csv` – columns **Delta**, **Gamma**, **Vega** |
| \(L_i\) | maximum long position of option \(i\) | `OptionCharacteristics.csv` – column **MaxLong** |
| \(S_i\) | minimum short position of option \(i\) (negative) | `OptionCharacteristics.csv` – column **MaxShort** |
| \(a_{ij}\) | binary reference of option \(i\) to asset \(j\) | `Option_AssetReferenceMatrix.csv` – columns **Asset_1**…**Asset_6** |
| \(\Delta^{0},\Gamma^{0},\nu^{0}\) | initial portfolio exposure | constants: 0.25, 0.08, 0.17 |
| \(\tau_{\Delta},\tau_{\Gamma},\tau_{\nu}\) | tolerance bands | constants: 0.06, 0.05, 0.07 |

Decision variables  

* \(x_i \in \mathbb{Z}\) number of contracts of option \(i\) (positive = long, negative = short)  
* \(y_i \ge 0\) auxiliary variable representing \(|x_i|\)

---

### Objective  

\[
\min \;\sum_{i\in I} c_i\, y_i
\]

### Constraints  

1. **Absolute‑value linearisation**  
   \[
   y_i \ge x_i,\qquad y_i \ge -x_i,\qquad \forall i\in I
   \]

2. **Trading limits**  
   \[
   S_i \;\le\; x_i \;\le\; L_i,\qquad \forall i\in I
   \]

3. **Risk exposure bounds**  

   For each Greek \(G\in\{\Delta,\Gamma,\nu\}\) let \(g_i\) denote the corresponding per‑contract Greek of option \(i\).  
   \[
   -\tau_G \;\le\; g^{0} + \sum_{i\in I}\sum_{j\in J} g_i\, a_{ij}\, x_i \;\le\; \tau_G,
   \qquad \forall G\in\{\Delta,\Gamma,\nu\}
   \]

4. **Variable domains**  
   \[
   x_i \in \mathbb{Z},\; y_i \ge 0,\qquad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(c_i\) | `file_0_view_0` | `Cost` |
| \(\Delta_i\) | `file_0_view_0` | `Delta` |
| \(\Gamma_i\) | `file_0_view_0` | `Gamma` |
| \(\nu_i\) | `file_0_view_0` | `Vega` |
| \(L_i\) | `file_0_view_0` | `MaxLong` |
| \(S_i\) | `file_0_view_0` | `MaxShort` |
| \(a_{ij}\) | `file_1_view_0` | `Asset_1` … `Asset_6` (columns 1–6) |
| \(\Delta^{0}\) | constants | 0.25 |
| \(\Gamma^{0}\) | constants | 0.08 |
| \(\nu^{0}\) | constants | 0.17 |
| \(\tau_{\Delta}\) | constants | 0.06 |
| \(\tau_{\Gamma}\) | constants | 0.05 |
| \(\tau_{\nu}\) | constants | 0.07 |

All indices, parameters, and constraints are defined solely from the data supplied in the observation and the problem statement. No additional entities or values are introduced.