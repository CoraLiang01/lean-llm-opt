**Mathematical Model**

Let  

* \(I=\{1,\dots ,11\}\) – set of potential warehouses  
* \(J=\{1,\dots ,11\}\) – set of stores  

Parameters  

\[
\begin{aligned}
f_i &\quad\text{opening cost of warehouse }i,\\
K_i &\quad\text{capacity of warehouse }i,\\
d_j &\quad\text{demand of store }j,\\
c_{ij}&\quad\text{transportation cost per unit from warehouse }i\text{ to store }j .
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(1 if warehouse }i\text{ is opened)} ,\\
x_{ij}&\ge 0\quad\text{(units shipped from }i\text{ to }j).
\end{aligned}
\]

**Objective**

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Capacity**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; K_i\,y_i \qquad \forall i\in I
   \]

2. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}\;=\; d_j \qquad \forall j\in J
   \]

3. **Variable domains**  
   \[
   y_i\in\{0,1\}\;\;\forall i\in I,\qquad
   x_{ij}\ge 0\;\;\forall i\in I,\;j\in J
   \]

---

**Data Mapping**

| Parameter | Source Table | Column (or matrix entry) | Mapping |
|-----------|--------------|--------------------------|---------|
| \(f_i\) | `file_0_view_0` | `Opening Cost (fi)` | `f[i] = value` |
| \(K_i\) | `file_0_view_0` | `Capacity (units)` | `K[i] = value` |
| \(d_j\) | `file_1_view_0` | `Demand (units, dj)` | `d[j] = value` |
| \(c_{ij}\) | `file_2_view_0` | matrix cell at row `Store (j)` and column `Warehouse (i)` | `c[i,j] = value` |

*Indices*  
- Warehouse indices \(i\) correspond to the values in column `Warehouse (i)` of `file_0_view_0` (1–11).  
- Store indices \(j\) correspond to the values in column `Store (j)` of `file_1_view_0` (1–11).  

All parameters are taken directly from the CSV files; no numerical values are enumerated in the model.