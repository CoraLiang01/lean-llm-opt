**Mathematical Model**

Let  

* \(I=\{1,\dots,11\}\) – set of potential warehouses  
* \(J=\{1,\dots,11\}\) – set of stores  

Parameters  

\[
\begin{aligned}
f_i &\quad\text{opening cost of warehouse }i,\\
C_i &\quad\text{capacity of warehouse }i,\\
d_j &\quad\text{demand of store }j,\\
c_{ij}&\quad\text{transportation cost per unit from warehouse }i\text{ to store }j .
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(1 if warehouse }i\text{ is opened)} ,\\
x_{ij} &\in\mathbb{Z}_{+}\quad\text{(units shipped from }i\text{ to }j).
\end{aligned}
\]

---

**Objective**

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

---

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
   \]

2. **Warehouse capacity**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; C_i\,y_i ,\qquad \forall i\in I
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\;\ge\;0 ,\qquad \forall i\in I,\; j\in J
   \]

4. **Binary opening decision**  
   \[
   y_i\in\{0,1\},\qquad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(f_i\)   | `file_0_view_0` | `"Opening Cost (fi)"` |
| \(C_i\)   | `file_0_view_0` | `"Capacity (units)"` |
| \(d_j\)   | `file_1_view_0` | `"Demand (units, dj)"` |
| \(c_{ij}\)| `file_2_view_0` | matrix entry at row `"Unnamed: 0"` (store \(j\)), column `"Warehouse (i)"` |

*Index mapping*  
- Warehouse indices \(i\) correspond to the values in column `"Warehouse (i)"` of `file_0_view_0`.  
- Store indices \(j\) correspond to the values in column `"Store (j)"` of `file_1_view_0`.  
- The transportation cost matrix `file_2_view_0` aligns rows and columns with these indices via the `"Unnamed: 0"` row identifier and the warehouse column identifiers.