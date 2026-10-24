**Symbolic Model**

Let  

* \(I\) – set of suppliers (rows of *fixed_cost.csv*).  
* \(J\) – set of stores (rows of *demand.csv*).  

Parameters  

\[
\begin{aligned}
f_i &\quad\text{fixed cost of opening supplier }i,\\
d_j &\quad\text{demand of store }j,\\
c_{ij}&\quad\text{transportation cost per unit from supplier }i\text{ to store }j.
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in\{0,1\}\quad\text{(supplier }i\text{ is open)},\\
x_{ij} &\ge 0\quad\text{(units shipped from }i\text{ to }j).
\end{aligned}
\]

**Objective**

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
   \]

2. **Supplier activation (big‑M)**  
   \[
   x_{ij} \;\le\; d_j\,y_i \qquad \forall i\in I,\; j\in J
   \]

3. **Variable domains**  
   \[
   y_i \in \{0,1\}\;\;\forall i\in I,\qquad
   x_{ij}\ge 0\;\;\forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Source Table ID | Source Column | Row Identifier |
|-----------|-----------------|---------------|----------------|
| \(f_i\)   | `file_1_view_0` | `fixed_costs` | `Unnamed: 0` (supplier name) |
| \(d_j\)   | `file_0_view_0` | `demand`      | `Customer` (store name) |
| \(c_{ij}\)| `file_2_view_0` | matrix cells  | Row: `Unnamed: 0` (supplier name); Column: store names (`CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`) |

*No numerical values are listed; each parameter is linked to its exact source table and column as shown.*