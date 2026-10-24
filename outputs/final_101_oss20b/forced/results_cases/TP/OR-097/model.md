**Mathematical Model**

Let  

* \(I=\{\,\text{Opt}_1,\dots ,\text{Opt}_{120}\,\}\) (120 options)  
* \(J=\{\,\text{Asset}_1,\dots ,\text{Asset}_6\,\}\) (6 underlying assets)

Parameters (all real numbers)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(\text{Cost}[i]\) | per‑contract cost of option \(i\) | `file_0_view_0`, column **Cost** |
| \(\Delta[i]\) | per‑contract Delta of option \(i\) | `file_0_view_0`, column **Delta** |
| \(\Gamma[i]\) | per‑contract Gamma of option \(i\) | `file_0_view_0`, column **Gamma** |
| \(\text{Vega}[i]\) | per‑contract Vega of option \(i\) | `file_0_view_0`, column **Vega** |
| \(\text{MaxLong}[i]\) | maximum long position for option \(i\) | `file_0_view_0`, column **MaxLong** |
| \(\text{MaxShort}[i]\) | maximum short position for option \(i\) (negative) | `file_0_view_0`, column **MaxShort** |
| \(A[i,j]\) | binary reference matrix (1 if option \(i\) references asset \(j\)) | `file_1_view_0`, column **Asset\_j** |
| \(\Delta_{\text{init}}\) | initial portfolio Delta | constant \(0.25\) |
| \(\Gamma_{\text{init}}\) | initial portfolio Gamma | constant \(0.08\) |
| \(\text{Vega}_{\text{init}}\) | initial portfolio Vega | constant \(0.17\) |
| \(\text{Tol}_\Delta\) | Delta tolerance | constant \(0.06\) |
| \(\text{Tol}_\Gamma\) | Gamma tolerance | constant \(0.05\) |
| \(\text{Tol}_{\text{Vega}}\) | Vega tolerance | constant \(0.07\) |

Decision variables

* \(x[i]\in\mathbb{Z}\) – number of contracts of option \(i\) (positive = long, negative = short)
* \(y[i]\ge 0\) – auxiliary variable representing \(|x[i]|\)
* \(g_\Delta,\; g_\Gamma,\; g_{\text{Vega}}\) – net portfolio exposures after hedging

---

### Objective
\[
\min \;\sum_{i\in I} \text{Cost}[i]\; y[i]
\]

### Constraints

1. **Absolute‑value linearisation for the objective**  
   \[
   -y[i] \;\le\; x[i] \;\le\; y[i] \qquad \forall i\in I
   \]

2. **Trading limits**  
   \[
   \text{MaxShort}[i] \;\le\; x[i] \;\le\; \text{MaxLong}[i] \qquad \forall i\in I
   \]

3. **Net Delta exposure**  
   \[
   g_\Delta \;=\; \Delta_{\text{init}} \;+\; \sum_{i\in I}\sum_{j\in J} \Delta[i]\;A[i,j]\;x[i]
   \]
   \[
   -\text{Tol}_\Delta \;\le\; g_\Delta \;\le\; \text{Tol}_\Delta
   \]

4. **Net Gamma exposure**  
   \[
   g_\Gamma \;=\; \Gamma_{\text{init}} \;+\; \sum_{i\in I}\sum_{j\in J} \Gamma[i]\;A[i,j]\;x[i]
   \]
   \[
   -\text{Tol}_\Gamma \;\le\; g_\Gamma \;\le\; \text{Tol}_\Gamma
   \]

5. **Net Vega exposure**  
   \[
   g_{\text{Vega}} \;=\; \text{Vega}_{\text{init}} \;+\; \sum_{i\in I}\sum_{j\in J} \text{Vega}[i]\;A[i,j]\;x[i]
   \]
   \[
   -\text{Tol}_{\text{Vega}} \;\le\; g_{\text{Vega}} \;\le\; \text{Tol}_{\text{Vega}}
   \]

All variables \(x[i]\) are integers; \(y[i], g_\Delta, g_\Gamma, g_{\text{Vega}}\) are continuous.

---

## Data Mapping Rules

| Parameter | Table ID | Column | Notes |
|-----------|----------|--------|-------|
| \(\text{Cost}[i]\) | `file_0_view_0` | `Cost` | per‑contract cost |
| \(\Delta[i]\) | `file_0_view_0` | `Delta` | per‑contract Delta |
| \(\Gamma[i]\) | `file_0_view_0` | `Gamma` | per‑contract Gamma |
| \(\text{Vega}[i]\) | `file_0_view_0` | `Vega` | per‑contract Vega |
| \(\text{MaxLong}[i]\) | `file_0_view_0` | `MaxLong` | maximum long limit |
| \(\text{MaxShort}[i]\) | `file_0_view_0` | `MaxShort` | maximum short limit (negative) |
| \(A[i,j]\) | `file_1_view_0` | `Asset_j` | binary reference (1 if option \(i\) references asset \(j\)) |
| \(\Delta_{\text{init}}\) | constant | – | 0.25 |
| \(\Gamma_{\text{init}}\) | constant | – | 0.08 |
| \(\text{Vega}_{\text{init}}\) | constant | – | 0.17 |
| \(\text{Tol}_\Delta\) | constant | – | 0.06 |
| \(\text{Tol}_\Gamma\) | constant | – | 0.05 |
| \(\text{Tol}_{\text{Vega}}\) | constant | – | 0.07 |

Indices \(i\) and \(j\) are mapped directly to the option identifiers (`Option` column in `file_0_view_0`) and asset identifiers (`Asset_1`–`Asset_6` columns in `file_1_view_0`). No aggregation or transformation of data is performed beyond the linearisation of absolute values.