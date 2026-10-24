**Mathematical Model**

Let  

- \(I=\{1,\dots,11\}\) be the set of potential warehouses.  
- \(J=\{1,\dots,11\}\) be the set of stores.

Parameters  

\[
\begin{aligned}
f_i &\quad\text{opening cost of warehouse }i,\\
C_i &\quad\text{capacity of warehouse }i,\\
d_j &\quad\text{demand of store }j,\\
c_{ij}&\quad\text{transportation cost per unit from warehouse }i\text{ to store }j.
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(1 if warehouse }i\text{ is opened)},\\
x_{ij}&\ge 0\quad\text{(units shipped from }i\text{ to }j).
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Capacity limits**

\[
\sum_{j\in J} x_{ij}\;\le\; C_i\,y_i \qquad \forall i\in I
\]

2. **Demand satisfaction**

\[
\sum_{i\in I} x_{ij}\;=\; d_j \qquad \forall j\in J
\]

3. **Non‑negativity**

\[
x_{ij}\;\ge\;0 \qquad \forall i\in I,\; j\in J
\]

4. **Binary opening decisions**

\[
y_i\in\{0,1\}\qquad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) |
|-----------|--------------|-----------|
| \(f_i\) | `file_0_view_0` | `Opening Cost (fi)` |
| \(C_i\) | `file_0_view_0` | `Capacity (units)` |
| \(d_j\) | `file_1_view_0` | `Demand (units, dj)` |
| \(c_{ij}\) | `file_2_view_0` | `W1`–`W11` columns (row header `Unnamed: 0` gives warehouse index) |

*Index mapping:*  
Warehouse index \(i\) corresponds to the value in column `Warehouse (i)` of `file_0_view_0`.  
Store index \(j\) corresponds to the value in column `Store (j)` of `file_1_view_0`.  
Transportation cost \(c_{ij}\) is read from the cell where row header `Unnamed: 0` equals `W{i}` and column header equals `W{j}` in `file_2_view_0`.