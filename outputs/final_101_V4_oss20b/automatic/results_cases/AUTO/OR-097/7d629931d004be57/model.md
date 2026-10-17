**Mathematical Model**

Let  

* \(I=\{1,\dots ,120\}\) – set of option indices (identified by the *Option* column).  
* \(J=\{1,\dots ,6\}\) – set of asset indices (columns *Asset_1* … *Asset_6*).  

Parameters  

\[
\begin{aligned}
&\text{Cost}[i] &&\text{(per‑contract cost)}\\
&\Delta[i],\;\Gamma[i],\;\text{Vega}[i] &&\text{(per‑contract Greek exposures)}\\
&\text{MaxLong}[i],\;\text{MaxShort}[i] &&\text{(trading limits)}\\
&A[i,j] &&\text{(binary reference matrix)}\\
&\Delta_0,\;\Gamma_0,\;\text{Vega}_0 &&\text{(initial exposures)}\\
&\text{Tol}_\Delta,\;\text{Tol}_\Gamma,\;\text{Tol}_{\text{Vega}} &&\text{(risk band tolerances)}
\end{aligned}
\]

Decision variables  

\[
x_i \in \mathbb{Z}\qquad \forall i\in I
\]

Objective  

\[
\min \;\sum_{i\in I}\text{Cost}[i]\;|x_i|
\]

Constraints  

1. **Trading limits**  
\[
\text{MaxShort}[i]\;\le\;x_i\;\le\;\text{MaxLong}[i]\qquad \forall i\in I
\]

2. **Delta band**  
\[
\Bigl|\;\Delta_0+\sum_{i\in I}\sum_{j\in J}\Delta[i]\;A[i,j]\;x_i\;\Bigr|\;\le\;\text{Tol}_\Delta
\]

3. **Gamma band**  
\[
\Bigl|\;\Gamma_0+\sum_{i\in I}\sum_{j\in J}\Gamma[i]\;A[i,j]\;x_i\;\Bigr|\;\le\;\text{Tol}_\Gamma
\]

4. **Vega band**  
\[
\Bigl|\;\text{Vega}_0+\sum_{i\in I}\sum_{j\in J}\text{Vega}[i]\;A[i,j]\;x_i\;\Bigr|\;\le\;\text{Tol}_{\text{Vega}}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| `Cost[i]` | `file_0_view_0` | `Cost` |
| `Delta[i]` | `file_0_view_0` | `Delta` |
| `Gamma[i]` | `file_0_view_0` | `Gamma` |
| `Vega[i]` | `file_0_view_0` | `Vega` |
| `MaxLong[i]` | `file_0_view_0` | `MaxLong` |
| `MaxShort[i]` | `file_0_view_0` | `MaxShort` |
| `A[i,j]` | `file_1_view_0` | `Asset_1` … `Asset_6` (columns 1–6) |
| `Delta_0` | constant | 0.25 |
| `Gamma_0` | constant | 0.08 |
| `Vega_0` | constant | 0.17 |
| `Tol_Δ` | constant | 0.06 |
| `Tol_Γ` | constant | 0.05 |
| `Tol_Vega` | constant | 0.07 |

All indices `i` and `j` are taken from the rows and columns of the respective tables; no additional sorting or renumbering is performed. The model uses the absolute‑value notation as specified in the problem statement.