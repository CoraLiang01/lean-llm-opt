**Mathematical Model**

Let  

* \(I\) – set of suppliers (indexed by \(i\)).  
* \(J\) – set of stores (indexed by \(j\)).  

Parameters  

\[
\begin{aligned}
&\text{fixed\_cost}_i &&\text{(fixed cost of opening supplier }i)\\
&\text{demand}_j &&\text{(demand of store }j)\\
&\text{transport}_{ij} &&\text{(unit transportation cost from supplier }i\text{ to store }j)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_i &\in\{0,1\} &&\text{(supplier }i\text{ is open)}\\
&x_{ij} &\ge 0 &&\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

---

### Objective  

\[
\min \;\; \sum_{i\in I}\text{fixed\_cost}_i\,y_i
      \;+\; \sum_{i\in I}\sum_{j\in J}\text{transport}_{ij}\,x_{ij}
\]

### Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; \text{demand}_j \qquad \forall j\in J
   \]

2. **Supplier activation** (big‑\(M\) formulation)  
   \[
   x_{ij} \;\le\; M\,y_i \qquad \forall i\in I,\; j\in J
   \]
   where \(M\) is a sufficiently large constant (e.g., the sum of all demands).

3. **Variable domains**  
   \[
   y_i \in \{0,1\}\;\;\forall i\in I,\qquad
   x_{ij}\ge 0\;\;\forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Index Mapping |
|-----------|--------------|---------------|---------------|
| \(\text{fixed\_cost}_i\) | `file_1_view_0` | `fixed_costs` | `i` ← `Unnamed: 0` |
| \(\text{demand}_j\) | `file_0_view_0` | `demand` | `j` ← `Customer` |
| \(\text{transport}_{ij}\) | `file_2_view_0` | `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT` | `i` ← `Unnamed: 0`, `j` ← column name |

*All numeric values are taken directly from the corresponding CSV files; no enumeration is shown.*