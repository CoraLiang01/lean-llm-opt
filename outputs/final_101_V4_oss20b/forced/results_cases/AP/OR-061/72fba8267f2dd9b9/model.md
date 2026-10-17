**Mathematical Model**

Let  

* \(I=\{S_1,S_2,S_3,S_4,S_5\}\) – set of suppliers  
* \(J=\{C_1,C_2,C_3,C_4,C_5\}\) – set of branches (customers)

Parameters  

\[
\begin{aligned}
&f_i &&\text{fixed cost of opening supplier }i, && i\in I\\
&d_j &&\text{demand of branch }j, && j\in J\\
&c_{ij} &&\text{transportation cost per unit from supplier }i\text{ to branch }j, && i\in I,\;j\in J
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_i &\in\{0,1\} &&\text{1 if supplier }i\text{ is opened, 0 otherwise}\\
&x_{ij} &\ge 0 &&\text{quantity shipped from supplier }i\text{ to branch }j
\end{aligned}
\]

---

### Objective  

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

### Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;=\; d_j \quad \forall j\in J
   \]

2. **Supplier activation** (no shipment if supplier closed)  
   \[
   x_{ij} \;\le\; d_j\,y_i \quad \forall i\in I,\; j\in J
   \]

3. **Variable domains**  
   \[
   y_i \in \{0,1\}\;\;\forall i\in I,\qquad
   x_{ij}\ge 0\;\;\forall i\in I,\;j\in J
   \]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column | Index Mapping |
|-----------|--------------|------------|--------------|---------------|
| \(f_i\)   | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` | \(i =\) supplier ID (`S1`–`S5`) |
| \(d_j\)   | `file_0_view_0` | `customer` | `demand` | \(j =\) branch ID (`C1`–`C5`) |
| \(c_{ij}\)| `file_2_view_0` | `Unnamed: 0` | `C1`–`C5` | \(i =\) supplier ID, \(j =\) branch ID |

All numerical values are taken directly from the corresponding CSV files; no enumeration of the table rows is shown in the model.