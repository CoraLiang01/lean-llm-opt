**Mathematical Model**

Let  

- \(I=\{S_1,\dots ,S_8\}\) suppliers  
- \(J=\{C_1,\dots ,C_9\}\) dealerships  

Parameters  

\[
\begin{aligned}
f_i &\;=\;\text{fixed cost of opening supplier }i, && i\in I\\
d_j &\;=\;\text{demand of dealership }j, && j\in J\\
c_{ij}&\;=\;\text{transportation cost per vehicle from supplier }i\text{ to dealership }j, && i\in I,\;j\in J\\
M &\;=\;\sum_{j\in J} d_j \quad(\text{big‑M upper bound on total shipments})
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\} &&\text{(supplier }i\text{ is open)}\\
x_{ij} &\ge 0 &&\text{(vehicles shipped from }i\text{ to }j)
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
   \]

2. **Supplier activation**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i ,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\}\quad \forall i\in I,\;j\in J
   \]

---

**Data Mapping**

| Parameter | Source File | Source Columns | Mapping |
|-----------|-------------|----------------|---------|
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv` | `Unnamed: 0` (supplier ID), `fixed_costs` | \(f_{\,\text{row supplier}}\) |
| \(d_j\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv` | `customer` (dealership ID), `demand` | \(d_{\,\text{row customer}}\) |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv` | `Unnamed: 0` (supplier ID), columns `C1`–`C9` (dealership IDs) | \(c_{\,\text{row supplier},\,\text{column dealership}}\) |

All numerical values are taken directly from the corresponding CSV files; no additional limits or capacities are imposed beyond the big‑\(M\) bound.