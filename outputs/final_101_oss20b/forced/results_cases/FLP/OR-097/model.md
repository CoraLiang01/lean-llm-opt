**Mathematical Model**

Let  

* \(I=\{\,\text{Opt}_1,\dots ,\text{Opt}_{120}\,\}\) (120 options)  
* \(J=\{\,\text{Asset}_1,\dots ,\text{Asset}_6\,\}\) (6 underlying assets)

Decision variables  
\[
x_i \in \mathbb{Z}\qquad \forall i\in I
\]

Parameters (to be instantiated from the data tables)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(\text{Cost}[i]\) | per‑contract cost of option \(i\) | `file_0_view_0.Cost` |
| \(\Delta[i]\) | per‑contract Delta of option \(i\) | `file_0_view_0.Delta` |
| \(\Gamma[i]\) | per‑contract Gamma of option \(i\) | `file_0_view_0.Gamma` |
| \(\text{Vega}[i]\) | per‑contract Vega of option \(i\) | `file_0_view_0.Vega` |
| \(\text{MaxLong}[i]\) | maximum long position for option \(i\) | `file_0_view_0.MaxLong` |
| \(\text{MaxShort}[i]\) | minimum short position for option \(i\) (negative) | `file_0_view_0.MaxShort` |
| \(A[i,j]\) | binary reference of option \(i\) to asset \(j\) | `file_1_view_0.Asset_j` |
| \(\Delta_{\text{init}}\) | initial portfolio Delta | 0.25 |
| \(\Gamma_{\text{init}}\) | initial portfolio Gamma | 0.08 |
| \(\text{Vega}_{\text{init}}\) | initial portfolio Vega | 0.17 |
| \(\Delta_{\text{tol}}\) | Delta tolerance | 0.06 |
| \(\Gamma_{\text{tol}}\) | Gamma tolerance | 0.05 |
| \(\text{Vega}_{\text{tol}}\) | Vega tolerance | 0.07 |

Objective  
\[
\min \;\sum_{i\in I}\text{Cost}[i]\;|x_i|
\]

Constraints  

1. **Delta band**  
\[
\bigl|\Delta_{\text{init}}+\sum_{i\in I}\sum_{j\in J}\Delta[i]\;A[i,j]\;x_i\bigr|\;\le\;\Delta_{\text{tol}}
\]

2. **Gamma band**  
\[
\bigl|\Gamma_{\text{init}}+\sum_{i\in I}\sum_{j\in J}\Gamma[i]\;A[i,j]\;x_i\bigr|\;\le\;\Gamma_{\text{tol}}
\]

3. **Vega band**  
\[
\bigl|\text{Vega}_{\text{init}}+\sum_{i\in I}\sum_{j\in J}\text{Vega}[i]\;A[i,j]\;x_i\bigr|\;\le\;\text{Vega}_{\text{tol}}
\]

4. **Trading limits**  
\[
\text{MaxShort}[i]\;\le\;x_i\;\le\;\text{MaxLong}[i]\qquad \forall i\in I
\]

5. **Integer domain**  
\[
x_i\in\mathbb{Z}\qquad \forall i\in I
\]

---

**Data Mapping Rules**

| Parameter | Table ID | Column | Mapping Rule |
|-----------|----------|--------|--------------|
| \(\text{Cost}[i]\) | `file_0_view_0` | `Cost` | For each row where `Option` = i, assign the value of `Cost`. |
| \(\Delta[i]\) | `file_0_view_0` | `Delta` | For each row where `Option` = i, assign the value of `Delta`. |
| \(\Gamma[i]\) | `file_0_view_0` | `Gamma` | For each row where `Option` = i, assign the value of `Gamma`. |
| \(\text{Vega}[i]\) | `file_0_view_0` | `Vega` | For each row where `Option` = i, assign the value of `Vega`. |
| \(\text{MaxLong}[i]\) | `file_0_view_0` | `MaxLong` | For each row where `Option` = i, assign the value of `MaxLong`. |
| \(\text{MaxShort}[i]\) | `file_0_view_0` | `MaxShort` | For each row where `Option` = i, assign the value of `MaxShort`. |
| \(A[i,j]\) | `file_1_view_0` | `Asset_j` | For each row where `Unnamed: 0` = i, assign the binary value of column `Asset_j` to \(A[i,j]\). |

All other constants (\(\Delta_{\text{init}},\Gamma_{\text{init}},\text{Vega}_{\text{init}},\Delta_{\text{tol}},\Gamma_{\text{tol}},\text{Vega}_{\text{tol}}\)) are fixed numerical values given in the problem statement.