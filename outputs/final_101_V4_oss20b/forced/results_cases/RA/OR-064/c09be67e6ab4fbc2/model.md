**Mathematical Model**

Let  

* \(I=\{i\mid\text{supplier }S_i\}\) – index set of suppliers (from *fixed_cost.csv*).  
* \(J=\{j\mid\text{customer }C_j\}\) – index set of supermarkets (from *demand.csv*).  

Decision variables  

\[
\begin{aligned}
y_i &\in \{0,1\} &&\text{(supplier \(S_i\) is open)}\\
x_{ij} &\ge 0 &&\text{(units supplied from \(S_i\) to \(C_j\))}
\end{aligned}
\]

Parameters  

\[
\begin{aligned}
f_i &\ge 0 &&\text{fixed cost of opening supplier }S_i\\
d_j &\ge 0 &&\text{demand of customer }C_j\\
c_{ij} &\ge 0 &&\text{transportation cost per unit from }S_i\text{ to }C_j
\end{aligned}
\]

---

### Objective  

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

### Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
   \]

2. **Supplier activation (big‑\(M\) formulation)**  
   \[
   x_{ij}\;\le\; d_j\,y_i ,\qquad \forall i\in I,\;\forall j\in J
   \]

3. **Variable domains**  
   \[
   y_i\in\{0,1\},\;\; x_{ij}\ge 0
   \]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(d_j\) | `file_0_view_0` | `demand` |
| \(c_{ij}\) | `file_2_view_0` | `Cj` (e.g., `C1`, `C2`, …, `C25`) |

*Supplier identifiers* are taken from the column `Unnamed: 0` in `file_1_view_0`.  
*Customer identifiers* are taken from the column `customer` in `file_0_view_0`.  

All indices, parameters, and constraints are defined solely from the supplied data and the problem statement.