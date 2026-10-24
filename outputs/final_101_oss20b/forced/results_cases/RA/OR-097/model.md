**Symbolic Model**

Let  

* \(I=\{1,\dots ,120\}\) index of options  
* \(J=\{1,\dots ,6\}\) index of assets  

Parameters (all taken from the supplied tables)

| Symbol | Definition | Source |
|--------|------------|--------|
| \( \text{Cost}[i] \) | per‑contract cost of option \(i\) | `file_0_view_0.Cost` |
| \( \Delta[i] \) | per‑contract Delta of option \(i\) | `file_0_view_0.Delta` |
| \( \Gamma[i] \) | per‑contract Gamma of option \(i\) | `file_0_view_0.Gamma` |
| \( \text{Vega}[i] \) | per‑contract Vega of option \(i\) | `file_0_view_0.Vega` |
| \( \text{MaxLong}[i] \) | maximum long position for option \(i\) | `file_0_view_0.MaxLong` |
| \( \text{MaxShort}[i] \) | minimum short position for option \(i\) | `file_0_view_0.MaxShort` |
| \( A[i,j] \) | binary indicator that option \(i\) references asset \(j\) | `file_1_view_0.Asset_j` (column `Asset_j` for row where `Unnamed: 0` = option \(i\)) |
| \( \Delta_0 \) | initial portfolio Delta | 0.25 |
| \( \Gamma_0 \) | initial portfolio Gamma | 0.08 |
| \( \text{Vega}_0 \) | initial portfolio Vega | 0.17 |
| \( \text{Tol}_\Delta \) | Delta tolerance | 0.06 |
| \( \text{Tol}_\Gamma \) | Gamma tolerance | 0.05 |
| \( \text{Tol}_{\text{Vega}} \) | Vega tolerance | 0.07 |

Decision variables  

\[
x_i \in \mathbb{Z}\quad\forall i\in I
\]

Objective  

\[
\min \;\sum_{i\in I} \text{Cost}[i]\;|x_i|
\]

Constraints  

1. **Delta band**

\[
\bigl|\;\Delta_0 + \sum_{i\in I}\sum_{j\in J}\Delta[i]\;A[i,j]\;x_i\;\bigr|\;\le\;\text{Tol}_\Delta
\]

2. **Gamma band**

\[
\bigl|\;\Gamma_0 + \sum_{i\in I}\sum_{j\in J}\Gamma[i]\;A[i,j]\;x_i\;\bigr|\;\le\;\text{Tol}_\Gamma
\]

3. **Vega band**

\[
\bigl|\;\text{Vega}_0 + \sum_{i\in I}\sum_{j\in J}\text{Vega}[i]\;A[i,j]\;x_i\;\bigr|\;\le\;\text{Tol}_{\text{Vega}}
\]

4. **Trading limits**

\[
\text{MaxShort}[i]\;\le\;x_i\;\le\;\text{MaxLong}[i]\qquad\forall i\in I
\]

All variables \(x_i\) are integers (positive for long, negative for short).

---

**Data Mapping Rules**

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| `Cost[i]` | `file_0_view_0` | `Cost` |
| `Delta[i]` | `file_0_view_0` | `Delta` |
| `Gamma[i]` | `file_0_view_0` | `Gamma` |
| `Vega[i]` | `file_0_view_0` | `Vega` |
| `MaxLong[i]` | `file_0_view_0` | `MaxLong` |
| `MaxShort[i]` | `file_0_view_0` | `MaxShort` |
| `A[i,j]` | `file_1_view_0` | `Asset_j` (where `j` = 1…6) |
| `Delta_0` | constant | 0.25 |
| `Gamma_0` | constant | 0.08 |
| `Vega_0` | constant | 0.17 |
| `Tol_Δ` | constant | 0.06 |
| `Tol_Γ` | constant | 0.05 |
| `Tol_Vega` | constant | 0.07 |

The mapping uses the option identifier in column `Option` (or `Unnamed: 0`) as the key for all option‑indexed parameters. No values are enumerated; the model references the full dataset via the specified table IDs and column names.