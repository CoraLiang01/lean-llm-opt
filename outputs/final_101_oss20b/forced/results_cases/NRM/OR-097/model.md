**Mathematical Model**

Let  

* \(I=\{1,\dots ,120\}\) – set of option indices (option identifiers).  
* \(J=\{1,\dots ,6\}\) – set of asset indices.

Parameters (all indexed by \(i\in I\) or \((i,j)\in I\times J\))  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(\text{Cost}[i]\) | per‑contract cost of option \(i\) | `file_0_view_0.Cost` |
| \(\Delta[i]\) | per‑contract Delta of option \(i\) | `file_0_view_0.Delta` |
| \(\Gamma[i]\) | per‑contract Gamma of option \(i\) | `file_0_view_0.Gamma` |
| \(\text{Vega}[i]\) | per‑contract Vega of option \(i\) | `file_0_view_0.Vega` |
| \(\text{MaxLong}[i]\) | maximum long position for option \(i\) | `file_0_view_0.MaxLong` |
| \(\text{MaxShort}[i]\) | minimum short position for option \(i\) | `file_0_view_0.MaxShort` |
| \(A[i,j]\) | binary reference matrix (1 if option \(i\) references asset \(j\)) | `file_1_view_0.Asset_k` where \(k=j\) |
| \(\Delta_0\) | initial portfolio Delta | \(0.25\) |
| \(\Gamma_0\) | initial portfolio Gamma | \(0.08\) |
| \(\text{Vega}_0\) | initial portfolio Vega | \(0.17\) |
| \(\text{Tol}_\Delta\) | Delta tolerance | \(0.06\) |
| \(\text{Tol}_\Gamma\) | Gamma tolerance | \(0.05\) |
| \(\text{Tol}_{\text{Vega}}\) | Vega tolerance | \(0.07\) |

Decision variables  

\[
x_i \in \mathbb{Z}\qquad \forall i\in I
\]

---

### Objective

\[
\min \;\sum_{i\in I} \text{Cost}[i]\;|x_i|
\]

---

### Constraints  

**1. Trading limits**

\[
\text{MaxShort}[i] \;\le\; x_i \;\le\; \text{MaxLong}[i] \qquad \forall i\in I
\]

**2. Risk‑band constraints (absolute‑value form)**  

For each Greek \(G\in\{\Delta,\Gamma,\text{Vega}\}\):

\[
\Bigl|\, G_0 \;+\; \sum_{i\in I}\sum_{j\in J} G[i]\;A[i,j]\;x_i \,\Bigr|
\;\le\; \text{Tol}_G
\]

Equivalently, split into two linear inequalities:

\[
\begin{aligned}
-\,\text{Tol}_G \;\le\; G_0 \;+\; \sum_{i\in I}\sum_{j\in J} G[i]\;A[i,j]\;x_i \;\le\; \text{Tol}_G
\end{aligned}
\qquad \forall G\in\{\Delta,\Gamma,\text{Vega}\}
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(\text{Cost}[i]\) | `file_0_view_0` | `Cost` |
| \(\Delta[i]\) | `file_0_view_0` | `Delta` |
| \(\Gamma[i]\) | `file_0_view_0` | `Gamma` |
| \(\text{Vega}[i]\) | `file_0_view_0` | `Vega` |
| \(\text{MaxLong}[i]\) | `file_0_view_0` | `MaxLong` |
| \(\text{MaxShort}[i]\) | `file_0_view_0` | `MaxShort` |
| \(A[i,1]\) | `file_1_view_0` | `Asset_1` |
| \(A[i,2]\) | `file_1_view_0` | `Asset_2` |
| \(A[i,3]\) | `file_1_view_0` | `Asset_3` |
| \(A[i,4]\) | `file_1_view_0` | `Asset_4` |
| \(A[i,5]\) | `file_1_view_0` | `Asset_5` |
| \(A[i,6]\) | `file_1_view_0` | `Asset_6` |

All other constants (\(\Delta_0,\Gamma_0,\text{Vega}_0,\text{Tol}_\Delta,\text{Tol}_\Gamma,\text{Tol}_{\text{Vega}}\)) are fixed values supplied in the problem statement.