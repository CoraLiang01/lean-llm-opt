**Mathematical Model**

Let  

* \(I=\{S1,S2,S3\}\) – set of warehouses  
* \(J=\{C1,C2,C3\}\) – set of customers  

Parameters  

\[
\begin{aligned}
\text{fixed\_cost}_i &\quad (i\in I)\\
\text{demand}_j &\quad (j\in J)\\
\text{transport}_{ij} &\quad (i\in I,\;j\in J)\\
M_j &= \text{demand}_j \qquad (j\in J)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in \{0,1\}\quad &&\text{(warehouse \(i\) is activated)}\\
x_{ij} &\ge 0 \quad &&\text{(units shipped from \(i\) to \(j\))}
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\text{fixed\_cost}_i\,y_i
      +\sum_{i\in I}\sum_{j\in J}\text{transport}_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; \text{demand}_j \qquad \forall j\in J
   \]

2. **Activation (big‑\(M\))**  
   \[
   x_{ij} \;\le\; M_j\,y_i \qquad \forall i\in I,\; j\in J
   \]

3. **Variable domains**  
   \[
   y_i \in \{0,1\}\;\;\forall i\in I,\qquad
   x_{ij}\ge 0\;\;\forall i\in I,\;j\in J
   \]

---

### Data Mapping

| Parameter | Table ID | Key column(s) | Value column(s) |
|-----------|----------|---------------|-----------------|
| `fixed_cost_i` | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| `demand_j` | `file_0_view_0` | `customer` | `demand` |
| `transport_{ij}` | `file_2_view_0` | `Unnamed: 0`, `customer` | `C1`, `C2`, `C3` (as appropriate for \(j\)) |
| `M_j` | *derived* | – | `demand_j` |

All indices and parameters are taken directly from the supplied CSV observations; no additional data are introduced.