**Mathematical Model**

Let  

* \(I\) – set of suppliers (indexed by \(i\)).  
* \(J\) – set of licensed stores (indexed by \(j\)).  

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
y_i &\in \{0,1\}\quad\text{(supplier }i\text{ is open)} ,\\
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
   \sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
   \]

2. **Supplier activation (big‑\(M\) formulation)**  
   \[
   x_{ij}\;\le\; d_j\,y_i ,\qquad \forall i\in I,\; \forall j\in J
   \]

3. **Variable domains**  
   \[
   y_i\in\{0,1\},\; x_{ij}\ge 0 ,\qquad \forall i\in I,\; \forall j\in J
   \]

---

### Data Mapping

| Parameter | Source | Table ID | Column(s) | Mapping rule |
|-----------|--------|----------|-----------|--------------|
| \(f_i\) | Fixed cost of suppliers | `file_1_view_0` | `fixed_costs` | \(f_i\) = value in row where `Unnamed: 0` = supplier \(i\) |
| \(d_j\) | Demand of stores | `file_0_view_0` | `demand` | \(d_j\) = value in row where `Customer` = store \(j\) |
| \(c_{ij}\) | Transportation cost | `file_2_view_0` | `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT` | \(c_{ij}\) = value in row where `Unnamed: 0` = supplier \(i\) and column = store \(j\) |

*All indices \(i\) and \(j\) are taken from the rows and columns of the respective tables.*