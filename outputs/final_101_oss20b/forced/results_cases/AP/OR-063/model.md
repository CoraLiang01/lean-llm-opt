**Mathematical Model**

Let  

* \(I=\{S_1,\dots ,S_7\}\) – set of warehouses (indexed by the value in column **Unnamed: 0** of *fixed_cost.csv*).  
* \(J=\{C_1,\dots ,C_7\}\) – set of customers (indexed by the value in column **customer** of *demand.csv*).  

Parameters  

\[
\begin{aligned}
f_i &\;=\;\text{fixed cost of warehouse }i \quad (i\in I)\\
t_{ij}&\;=\;\text{transportation cost per unit from warehouse }i\text{ to customer }j\\
d_j &\;=\;\text{demand of customer }j
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad &&\text{warehouse }i\text{ is activated}\\
x_{ij} &\ge 0 \quad &&\text{quantity shipped from }i\text{ to }j
\end{aligned}
\]

---

**Objective**

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

---

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}\;\ge\; d_j \qquad \forall j\in J
   \]

2. **Linking supply to activation**  
   \[
   x_{ij}\;\le\; d_j\,y_i \qquad \forall i\in I,\; j\in J
   \]

3. **Variable domains**  
   \[
   y_i\in\{0,1\}\;\;\forall i\in I,\qquad
   x_{ij}\ge 0\;\;\forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(f_i\)   | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| \(t_{ij}\)| `file_2_view_0` | `Unnamed: 0` | `C1`–`C7` (column name equals customer id) |
| \(d_j\)   | `file_0_view_0` | `customer` | `demand` |

*All indices \(i\) and \(j\) are taken from the corresponding key columns of the tables above.*