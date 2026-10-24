**Mathematical Model**

Let  

* \(I=\{i\mid i\in\text{file\_1\_view\_0.Unnamed: 0}\}\) – set of suppliers  
* \(J=\{j\mid j\in\text{file\_0\_view\_0.customer}\}\) – set of supermarkets  

Parameters  

\[
\begin{aligned}
f_i &:= \text{fixed cost of opening supplier }i
      &&\text{from } \text{file\_1\_view\_0.fixed\_costs} \\
d_j &:= \text{demand of supermarket }j
      &&\text{from } \text{file\_0\_view\_0.demand} \\
c_{ij} &:= \text{transportation cost per unit from supplier }i
          \text{ to supermarket }j
      &&\text{from } \text{file\_2\_view\_0.Cj}
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_i &\in \{0,1\} &&\text{(supplier }i\text{ is open)}\\
x_{ij} &\ge 0 &&\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

Objective  

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j \qquad \forall j\in J
   \]

2. **Supplier activation**  
   \[
   x_{ij}\le d_j\,y_i \qquad \forall i\in I,\; \forall j\in J
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\ge 0 \qquad \forall i\in I,\; \forall j\in J
   \]

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| \(d_j\) | `file_0_view_0` | `customer` | `demand` |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` (supplier) | `Cj` (customer column) |

*Index sets*  
- \(I\) is the set of all distinct values in `file_1_view_0.Unnamed: 0`.  
- \(J\) is the set of all distinct values in `file_0_view_0.customer`.  

All coefficients are taken directly from the corresponding CSV files; no values are enumerated or altered.