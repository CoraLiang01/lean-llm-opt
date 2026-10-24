**Mathematical Model**

Let  

* \(I=\{1,\dots ,11\}\) : set of potential warehouses  
* \(J=\{1,\dots ,11\}\) : set of stores  

Parameters  

\[
\begin{aligned}
f_i &\quad\text{opening cost of warehouse }i\\
c_{ij}&\quad\text{transportation cost per unit from warehouse }i\text{ to store }j\\
d_j &\quad\text{demand of store }j\\
\text{cap}_i &\quad\text{capacity of warehouse }i
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(1 if warehouse }i\text{ is opened)}\\
x_{ij}&\ge 0\quad\text{(units shipped from }i\text{ to }j)
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

2. **Capacity limits**  
   \[
   \sum_{j\in J} x_{ij}\le \text{cap}_i\,y_i ,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   y_i\in\{0,1\},\; x_{ij}\ge 0
   \]

---

**Data Mapping**

| Parameter | Source file | Source column(s) | Mapping rule |
|-----------|-------------|------------------|--------------|
| \(f_i\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv` | `Opening Cost (fi)` | \(f_i =\) value in row *i*, column `Opening Cost (fi)` |
| \(\text{cap}_i\) | same file | `Capacity (units)` | \(\text{cap}_i =\) value in row *i*, column `Capacity (units)` |
| \(d_j\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv` | `Demand (units, dj)` | \(d_j =\) value in row *j*, column `Demand (units, dj)` |
| \(c_{ij}\) | `Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv` | row header `W<i>`, column header `W<j>` | \(c_{ij} =\) value in row `W<i>`, column `W<j>` |

All indices \(i\) and \(j\) are taken from the row/column identifiers in the respective files (e.g., `W1`, `W2`, …, `W11` for warehouses and `1`, `2`, …, `11` for stores). No additional constraints or parameters are introduced beyond those explicitly required by the problem statement.