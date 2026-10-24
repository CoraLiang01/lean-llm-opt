**Mathematical Model**

Let  

* \(I=\{S_1,\dots ,S_5\}\) – set of suppliers (rows of *fixed_cost.csv* and *transportation_costs.csv*)  
* \(J=\{C_1,\dots ,C_5\}\) – set of branches (rows of *demand.csv* and columns of *transportation_costs.csv*)

Parameters  

\[
\begin{aligned}
&f_i &&\text{fixed cost of opening supplier }i \quad (i\in I)\\
&d_j &&\text{demand of branch }j \quad (j\in J)\\
&t_{ij} &&\text{transportation cost per unit from supplier }i\text{ to branch }j \quad (i\in I,\;j\in J)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_i &\in\{0,1\} &&\text{1 if supplier }i\text{ is opened, 0 otherwise}\\
&x_{ij} &\ge 0 &&\text{quantity shipped from supplier }i\text{ to branch }j
\end{aligned}
\]

**Objective**

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**

\[
\sum_{i\in I} x_{ij}=d_j \quad \forall j\in J
\]

2. **Supplier activation**

\[
x_{ij}\le d_j\,y_i \quad \forall i\in I,\; j\in J
\]

3. **Variable domains**

\[
y_i\in\{0,1\}\;\;\forall i\in I,\qquad
x_{ij}\ge 0\;\;\forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(f_i\) | *fixed_cost.csv* | `Unnamed: 0` → supplier id, `fixed_costs` → cost | \(f_{\,\text{row.Unnamed: 0}} = \text{row.fixed_costs}\) |
| \(d_j\) | *demand.csv* | `customer` → branch id, `demand` → quantity | \(d_{\,\text{row.customer}} = \text{row.demand}\) |
| \(t_{ij}\) | *transportation_costs.csv* | `Unnamed: 0` → supplier id, columns `C1`–`C5` → branch ids | \(t_{\,\text{row.Unnamed: 0},\,\text{col}} = \text{row.col}\) |

All indices \(i\) and \(j\) are taken from the exact identifiers present in the respective source tables; no additional entities are introduced. The model uses the full 5 × 5 transportation cost matrix and the 5 demand values as supplied.