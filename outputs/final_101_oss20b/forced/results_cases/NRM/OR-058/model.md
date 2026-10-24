**Symbolic Model**

\[
\begin{aligned}
\text{Minimize}\quad & \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{subject to}\quad
& \sum_{i\in I} x_{ij} \;\ge\; d_j , && \forall j\in J \qquad\text{(demand)}\\[4pt]
& x_{ij} \;\le\; M\,y_i , && \forall i\in I,\; j\in J \qquad\text{(linking)}\\[4pt]
& x_{ij} \;\ge\; 0 , && \forall i\in I,\; j\in J \\[4pt]
& y_i \;\in\{0,1\} , && \forall i\in I
\end{aligned}
\]

*Indices*  
\(I\) – set of suppliers (S1,…,S6)  
\(J\) – set of customers (C1,…,C6)  

*Parameters*  
\(f_i\) – fixed opening cost of supplier \(i\)  
\(d_j\) – demand of customer \(j\)  
\(c_{ij}\) – unit transportation cost from supplier \(i\) to customer \(j\)  
\(M\) – a sufficiently large constant (e.g., \(\sum_{j} d_j\))

*Variables*  
\(y_i\) – binary variable, 1 if supplier \(i\) is opened  
\(x_{ij}\) – quantity shipped from supplier \(i\) to customer \(j\)

---

**Data Mapping**

| Parameter | Source Table | Source Column | Index Column |
|-----------|--------------|---------------|--------------|
| \(f_i\)   | `file_1_view_0` | `fixed_costs` | `Unnamed: 0` (supplier id) |
| \(d_j\)   | `file_0_view_0` | `demand`      | `customer` (customer id) |
| \(c_{ij}\)| `file_2_view_0` | `C1`–`C6` (one column per customer) | `Unnamed: 0` (supplier id) |

*Note:* The mapping uses the exact table identifiers and column names provided in the observation. No numerical values are included in the model; they are to be supplied from the referenced tables.