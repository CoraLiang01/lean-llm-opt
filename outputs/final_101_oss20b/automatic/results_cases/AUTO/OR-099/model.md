**Mathematical model**

Let  

* \(I=\{1,\dots ,11\}\) : set of potential warehouses  
* \(J=\{1,\dots ,11\}\) : set of stores  

Parameters  

\[
\begin{aligned}
f_i &\;=\;\text{opening cost of warehouse }i, && i\in I\\
C_i &\;=\;\text{capacity of warehouse }i, && i\in I\\
d_j &\;=\;\text{demand of store }j, && j\in J\\
c_{ij}&\;=\;\text{transportation cost per unit from warehouse }i\text{ to store }j, && i\in I,\;j\in J
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\} &&\text{(warehouse }i\text{ is opened)}\\
x_{ij}&\ge 0 &&\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

Objective  

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
   \]

2. **Capacity & activation**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; C_i\,y_i ,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\}\quad \forall i\in I,\;j\in J
   \]

---

**Data‑mapping rules**

| Parameter | Source table | Source column / matrix entry | Mapping rule |
|-----------|--------------|------------------------------|--------------|
| \(I\) | `file_0_view_0` | `Warehouse (i)` | Set of all values in this column |
| \(J\) | `file_1_view_0` | `Store (j)` | Set of all values in this column |
| \(f_i\) | `file_0_view_0` | `Opening Cost (fi)` | For each row, map `Warehouse (i)` → `Opening Cost (fi)` |
| \(C_i\) | `file_0_view_0` | `Capacity (units)` | For each row, map `Warehouse (i)` → `Capacity (units)` |
| \(d_j\) | `file_1_view_0` | `Demand (units, dj)` | For each row, map `Store (j)` → `Demand (units, dj)` |
| \(c_{ij}\) | `file_2_view_0` | Matrix entry at row `j`, column `i` | For each row (store `j`) and each column (warehouse `i`), map the numeric value to \(c_{ij}\) |

*All indices \(i\) and \(j\) are taken from the corresponding sets \(I\) and \(J\) defined above.*