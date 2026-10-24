**Mathematical Model**

Let  

* \(I=\{i\mid i\text{ is a supplier in } \texttt{fixed\_cost.csv}\}\)  
* \(J=\{j\mid j\text{ is a supermarket in } \texttt{demand.csv}\}\)

Parameters  

\[
\begin{aligned}
f_i &\;=\;\text{fixed cost of supplier }i \quad (i\in I)\\
d_j &\;=\;\text{demand of supermarket }j \quad (j\in J)\\
c_{ij}&\;=\;\text{unit transportation cost from supplier }i\text{ to supermarket }j \quad (i\in I,\;j\in J)\\
M &\;=\;\sum_{j\in J} d_j \quad\text{(big‑\(M\) upper bound)}
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad &&\text{(supplier }i\text{ is activated)}\\
x_{ij} &\ge 0 \quad &&\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

Objective  

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j \quad \forall j\in J
   \]

2. **Supplier activation (big‑\(M\) formulation)**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i \quad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\qquad y_i\in\{0,1\}\quad \forall i\in I,\;j\in J
   \]

---

**Data Mapping**

| Parameter | Source Table | Row Axis | Column Axis | Mapping |
|-----------|--------------|----------|-------------|---------|
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | `f[i] = fixed_costs` |
| \(d_j\) | `file_0_view_0` | `customer` | `demand` | `d[j] = demand` |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` | `customer` | `c[i,j] = value` |
| \(M\) | – | – | – | `M = sum_j d[j]` |

*Index sets*  
- \(I\) is the set of values in column `Unnamed: 0` of `file_1_view_0`.  
- \(J\) is the set of values in column `customer` of `file_0_view_0`.

All parameters are taken directly from the CSV files; no numerical values are enumerated in the model.