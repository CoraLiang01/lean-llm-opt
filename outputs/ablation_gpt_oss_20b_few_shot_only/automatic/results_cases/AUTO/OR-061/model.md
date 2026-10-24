**Mathematical Model**

Let  

* \(I=\{S1,S2,S3,S4,S5\}\) – set of suppliers  
* \(J=\{C1,C2,C3,C4,C5\}\) – set of branches  

Parameters  

\[
\begin{aligned}
f_i &\quad\text{fixed cost of opening supplier }i, && i\in I\\
c_{ij}&\quad\text{transportation cost per unit from supplier }i\text{ to branch }j, && i\in I,\;j\in J\\
d_j &\quad\text{demand of branch }j, && j\in J\\
M &\quad\text{big‑\(M\) bound on total shipment from a supplier}
      =\sum_{j\in J} d_j
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\} &&\text{(supplier \(i\) is open)}\\
x_{ij} &\ge 0 &&\text{quantity shipped from supplier \(i\) to branch \(j\)}
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j \quad\forall j\in J
   \]

2. **Supplier activation**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i \quad\forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\qquad y_i\in\{0,1\}\quad\forall i\in I,\;j\in J
   \]

---

### Data Mapping

| Parameter | Source File | Source Column(s) | Mapping |
|-----------|-------------|------------------|---------|
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv` | `Unnamed: 0` (supplier ID), `fixed_costs` | \(f_i =\) value in column `fixed_costs` for row where `Unnamed: 0` = \(i\) |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv` | `Unnamed: 0` (supplier ID), columns `C1`–`C5` | \(c_{ij} =\) value in column `Cj` for row where `Unnamed: 0` = \(i\) |
| \(d_j\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv` | `customer`, `demand` | \(d_j =\) value in column `demand` for row where `customer` = \(j\) |
| \(M\) | Computed | – | \(M = \sum_{j\in J} d_j\) |

All indices, parameters, and variables are defined exactly as above; no numerical values are enumerated in the model.