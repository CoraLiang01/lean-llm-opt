**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & Z = \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{subject to}\quad
& \sum_{j\in J} x_{ij} \;\le\; s_i , && \forall i\in I \quad\text{(supply capacity)}\\[4pt]
& \sum_{i\in I} x_{ij} \;=\; d_j , && \forall j\in J \quad\text{(demand satisfaction)}\\[4pt]
& x_{ij}\;\ge\;0 , && \forall i\in I,\; j\in J
\end{aligned}
\]

**Index Sets**

- \(I\): set of suppliers (Suppliers 1–5)  
- \(J\): set of customers (Customer1–Customer6)

**Parameters**

- \(d_j\): demand of customer \(j\)  
- \(s_i\): supply capacity of supplier \(i\)  
- \(c_{ij}\): transportation cost per unit from supplier \(i\) to customer \(j\)

**Variables**

- \(x_{ij}\): quantity shipped from supplier \(i\) to customer \(j\) (continuous, \(\ge 0\))

---

### Data Mapping

| Parameter | Source Table | Source Column | Row Identifier |
|-----------|--------------|---------------|----------------|
| \(d_j\)   | `file_0_view_0` | `demand` | `Customers` |
| \(s_i\)   | `file_1_view_0` | `supply_capacity` | `Suppliers` |
| \(c_{ij}\)| `file_2_view_0` | Customer columns (`Customer1`–`Customer6`) | `Unnamed: 0` (Suppliers) |

All parameter values are taken directly from the corresponding CSV files; no additional data or assumptions are introduced.