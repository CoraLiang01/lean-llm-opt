**Mathematical Model**

Let  

* \(I\) – set of suppliers (indexed by \(i\)).  
* \(J\) – set of stores (indexed by \(j\)).  

Parameters  

\[
\begin{aligned}
f_i &\quad\text{fixed cost of opening supplier }i,\\
d_j &\quad\text{demand of store }j,\\
c_{ij}&\quad\text{transportation cost per unit from supplier }i\text{ to store }j,\\
M_i &\quad\text{big‑\(M\) for supplier }i\;(\text{e.g. }M_i=\sum_{j\in J}d_j).
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(supplier \(i\) is open)} ,\\
x_{ij} &\ge 0\quad\text{(units shipped from \(i\) to \(j\))}.
\end{aligned}
\]

---

### Objective

\[
\min \;\;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

---

### Constraints

1. **Demand satisfaction**

\[
\sum_{i\in I} x_{ij}=d_j \qquad \forall j\in J
\]

2. **Supplier activation linkage**

\[
x_{ij}\;\le\; M_i\,y_i \qquad \forall i\in I,\; j\in J
\]

3. **Variable domains**

\[
y_i\in\{0,1\}\;\;\forall i\in I,\qquad
x_{ij}\ge 0\;\;\forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | `i` ← value of `Unnamed: 0` (supplier name) |
| \(d_j\) | `file_0_view_0` | `demand` | `j` ← value of `Customer` (store name) |
| \(c_{ij}\) | `file_2_view_0` | `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT` | `i` ← value of `Unnamed: 0` (supplier name); `j` ← column name (store name) |
| \(M_i\) | – | – | \(M_i = \sum_{j\in J} d_j\) (computed from demand data) |

All indices \(i\) and \(j\) are taken from the complete sets of suppliers and stores present in the respective CSV files. No additional entities are introduced.