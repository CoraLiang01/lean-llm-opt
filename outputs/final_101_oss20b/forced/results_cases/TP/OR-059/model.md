**Mathematical model**

Let  

* \(I=\{S1,\dots ,S8\}\) suppliers  
* \(J=\{C1,\dots ,C4\}\) dealerships  

Parameters  

\[
\begin{aligned}
&\text{fixed\_cost}_i &&\text{(fixed cost of opening supplier }i)\\
&\text{demand}_j &&\text{(vehicles required by dealership }j)\\
&c_{ij} &&\text{(transportation cost per vehicle from supplier }i\text{ to dealership }j)\\
&M &&\text{big‑M, e.g. }M=\sum_{j\in J}\text{demand}_j
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_i &\in\{0,1\} &&\text{(supplier }i\text{ is open)}\\
&x_{ij}&\ge 0 &&\text{(vehicles shipped from }i\text{ to }j)
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\text{fixed\_cost}_i\,y_i
      +\sum_{i\in I}\sum_{j\in J}c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I}x_{ij}= \text{demand}_j \qquad \forall j\in J
   \]

2. **Supplier activation (big‑M)**  
   \[
   x_{ij}\le M\,y_i \qquad \forall i\in I,\; j\in J
   \]

3. **Variable domains**  
   \[
   y_i\in\{0,1\}\;\;\forall i\in I,\qquad
   x_{ij}\ge 0\;\;\forall i\in I,\; j\in J
   \]

---

### Data‑mapping rules

| Parameter | Source table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| `fixed_cost_i` | `file_1_view_0` | `fixed_costs` | Index `i` comes from column `Unnamed: 0` |
| `demand_j` | `file_0_view_0` | `demand` | Index `j` comes from column `customer` |
| `c_{ij}` | `file_2_view_0` | `C1`–`C4` | Row index `i` from column `Unnamed: 0`; column names give `j` |
| `M` | computed | – | `M = Σ_j demand_j` (can be pre‑computed or set to a sufficiently large constant) |

All indices, parameters, and variables are defined exactly as above; no additional data are invented or omitted.