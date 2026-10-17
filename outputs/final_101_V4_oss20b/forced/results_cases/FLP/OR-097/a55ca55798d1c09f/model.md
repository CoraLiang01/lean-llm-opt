**Mathematical Model**

Let  

* \(I=\{1,\dots ,120\}\) – index set of options (option \(i\) is identified by the
  *Option* column in *OptionCharacteristics.csv*).  
* \(J=\{1,\dots ,6\}\) – index set of assets (asset \(j\) corresponds to column
  *Asset\_j* in *Option_AssetReferenceMatrix.csv*).

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(Cost_i\) | per‑contract cost of option \(i\) | `file_0_view_0.Cost` |
| \(\Delta_i\) | per‑contract Delta of option \(i\) | `file_0_view_0.Delta` |
| \(\Gamma_i\) | per‑contract Gamma of option \(i\) | `file_0_view_0.Gamma` |
| \(\Vega_i\) | per‑contract Vega of option \(i\) | `file_0_view_0.Vega` |
| \(MaxLong_i\) | maximum long position for option \(i\) | `file_0_view_0.MaxLong` |
| \(MaxShort_i\) | maximum short position for option \(i\) | `file_0_view_0.MaxShort` |
| \(A_{ij}\) | binary reference of option \(i\) to asset \(j\) | `file_1_view_0.Asset_j` (row where *Unnamed: 0* = option \(i\)) |
| \(\Delta^{init}\) | initial portfolio Delta | 0.25 |
| \(\Gamma^{init}\) | initial portfolio Gamma | 0.08 |
| \(\Vega^{init}\) | initial portfolio Vega | 0.17 |
| \(Tol_{\Delta}\) | Delta tolerance | 0.06 |
| \(Tol_{\Gamma}\) | Gamma tolerance | 0.05 |
| \(Tol_{\Vega}\) | Vega tolerance | 0.07 |

**Decision Variables**

* \(x_i \in \mathbb{Z}\) – number of contracts of option \(i\) (positive = long,
  negative = short).

**Objective**

\[
\min \;\sum_{i\in I} Cost_i \, |x_i|
\]

**Constraints**

1. **Greek exposure limits**

\[
\begin{aligned}
\bigl|\Delta^{init} + \sum_{i\in I}\sum_{j\in J}\Delta_i\,A_{ij}\,x_i\bigr| &\le Tol_{\Delta},\\[4pt]
\bigl|\Gamma^{init} + \sum_{i\in I}\sum_{j\in J}\Gamma_i\,A_{ij}\,x_i\bigr| &\le Tol_{\Gamma},\\[4pt]
\bigl|\Vega^{init} + \sum_{i\in I}\sum_{j\in J}\Vega_i\,A_{ij}\,x_i\bigr| &\le Tol_{\Vega}.
\end{aligned}
\]

2. **Trading limits**

\[
MaxShort_i \;\le\; x_i \;\le\; MaxLong_i , \qquad \forall i\in I .
\]

3. **Integer domain**

\[
x_i \in \mathbb{Z}, \qquad \forall i\in I .
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| \(Cost_i\) | `file_0_view_0` | `Cost` |
| \(\Delta_i\) | `file_0_view_0` | `Delta` |
| \(\Gamma_i\) | `file_0_view_0` | `Gamma` |
| \(\Vega_i\) | `file_0_view_0` | `Vega` |
| \(MaxLong_i\) | `file_0_view_0` | `MaxLong` |
| \(MaxShort_i\) | `file_0_view_0` | `MaxShort` |
| \(A_{ij}\) | `file_1_view_0` | `Asset_1` … `Asset_6` (column *Asset_j*) |
| \(\Delta^{init}\) | – | constant 0.25 |
| \(\Gamma^{init}\) | – | constant 0.08 |
| \(\Vega^{init}\) | – | constant 0.17 |
| \(Tol_{\Delta}\) | – | constant 0.06 |
| \(Tol_{\Gamma}\) | – | constant 0.05 |
| \(Tol_{\Vega}\) | – | constant 0.07 |

*Indices \(i\) and \(j\) are mapped to the row and column identifiers in the
respective tables as shown above.*