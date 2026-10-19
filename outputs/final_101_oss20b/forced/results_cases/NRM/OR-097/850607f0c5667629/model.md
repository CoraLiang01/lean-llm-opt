##### Abstract Model

**Index Sets**  
\[
I=\{1,2,\dots ,120\}, \qquad J=\{1,2,\dots ,6\}
\]

**Parameters**  

\[
\begin{aligned}
&\text{Cost}[i] &&\text{(per‑contract cost of option }i)\\
&\Delta[i] &&\text{(per‑contract delta of option }i)\\
&\Gamma[i] &&\text{(per‑contract gamma of option }i)\\
&\text{Vega}[i] &&\text{(per‑contract vega of option }i)\\
&\text{MaxLong}[i] &&\text{(maximum long position allowed for option }i)\\
&\text{MaxShort}[i] &&\text{(maximum short position allowed for option }i)\\
& A[i,j] &&\begin{cases}
1 & \text{if option }i\text{ references asset }j\\
0 & \text{otherwise}
\end{cases}\\
&\Delta_{\text{init}} &&0.25\\
&\Gamma_{\text{init}} &&0.08\\
&\text{Vega}_{\text{init}} &&0.17\\
&T_{\Delta} &&0.06\\
&T_{\Gamma} &&0.05\\
&T_{\text{Vega}} &&0.07
\end{aligned}
\]

**Decision Variables**  

\[
x[i]\in \mathbb{Z}\quad\forall i\in I
\]

**Objective Function**  

\[
\min \;\sum_{i\in I}\text{Cost}[i]\;\bigl|x[i]\bigr|
\]

**Constraints**

1. **Delta Exposure Band**  
\[
\bigl|\Delta_{\text{init}}+\sum_{i\in I}\sum_{j\in J}\Delta[i]\;A[i,j]\;x[i]\bigr|\;\leq\;T_{\Delta}
\]

2. **Gamma Exposure Band**  
\[
\bigl|\Gamma_{\text{init}}+\sum_{i\in I}\sum_{j\in J}\Gamma[i]\;A[i,j]\;x[i]\bigr|\;\leq\;T_{\Gamma}
\]

3. **Vega Exposure Band**  
\[
\bigl|\text{Vega}_{\text{init}}+\sum_{i\in I}\sum_{j\in J}\text{Vega}[i]\;A[i,j]\;x[i]\bigr|\;\leq\;T_{\text{Vega}}
\]

4. **Trading Limits**  
\[
\text{MaxShort}[i]\;\leq\;x[i]\;\leq\;\text{MaxLong}[i]\quad\forall i\in I
\]

5. **Integer Domain**  
\[
x[i]\in \mathbb{Z}\quad\forall i\in I
\]

---

#### Data Mapping

| Parameter | Source Table | Exact Column Name |
|-----------|--------------|-------------------|
| `Cost[i]` | `file_0_view_0` | `Cost` |
| `Delta[i]` | `file_0_view_0` | `Delta` |
| `Gamma[i]` | `file_0_view_0` | `Gamma` |
| `Vega[i]` | `file_0_view_0` | `Vega` |
| `MaxLong[i]` | `file_0_view_0` | `MaxLong` |
| `MaxShort[i]` | `file_0_view_0` | `MaxShort` |
| `A[i,j]` | `file_1_view_0` | `Asset_1` … `Asset_6` (column `Asset_k` corresponds to asset \(k\)) |

All other constants (`Δ_init`, `Γ_init`, `Vega_init`, `T_Δ`, `T_Γ`, `T_Vega`) are fixed values given in the problem statement.