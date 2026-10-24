**Mathematical Model**

Let  

* \(I=\{i\mid i\text{ is a supplier in }{\tt fixed\_cost.csv}\}\)  
* \(J=\{j\mid j\text{ is a customer in }{\tt demand.csv}\}\)

Parameters  

\[
\begin{aligned}
f_i &\;=\;\text{fixed cost of opening supplier }i,\\
c_{ij}&\;=\;\text{transportation cost per unit from supplier }i\text{ to customer }j,\\
d_j &\;=\;\text{demand of customer }j,\\
M &\;=\;\sum_{j\in J} d_j .
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
x_{ij}&\ge 0 &&\text{quantity shipped from }i\text{ to }j,\\
y_i&\in\{0,1\} &&\text{indicator that supplier }i\text{ is opened}.
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}\;+\;\sum_{i\in I} f_i\,y_i .
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j \qquad \forall j\in J .
   \]

2. **Supplier activation**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i \qquad \forall i\in I .
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\}\qquad \forall i\in I,\;j\in J .
   \]

---

### Data‑Mapping Rules

| Parameter | Source Table | Row Identifier | Column Identifier |
|-----------|--------------|----------------|-------------------|
| \(f_i\)   | `file_1_view_0` | `Unnamed: 0` (supplier name) | `fixed_costs` |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` (supplier name) | **customer id** \(j\) (e.g., `Customer_1`, `Customer_2`, …) |
| \(d_j\)   | `file_0_view_0` | `Customer` (customer id) | `demand` |
| \(M\)     | – | – | – (computed as \(\sum_j d_j\)) |

*Index sets*  
- \(I\) is the set of all values appearing in the `Unnamed: 0` column of `file_1_view_0`.  
- \(J\) is the set of all values appearing in the `Customer` column of `file_0_view_0`.

These mappings bind each symbolic parameter directly to the corresponding cell in the CSV data. No numerical values are enumerated in the model.