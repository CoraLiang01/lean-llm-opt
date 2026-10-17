**Symbolic Model**

Let  

- \(I\) be the set of products whose *Product Name* contains the string “27in”.  
- For each \(i\in I\)  

  \[
  \begin{aligned}
  R_i &\;=\;\text{Revenue of product }i,\\
  D_i &\;=\;\text{Demand of product }i,\\
  S_i &\;=\;\text{Initial Inventory of product }i.
  \end{aligned}
  \]

Decision variable  

\[
x_i \;\in\; \mathbb{Z}_{+}\qquad \forall i\in I
\]

(Units of product \(i\) to be sold).

---

**Objective**

\[
\max \;\sum_{i\in I} R_i\,x_i
\]

---

**Constraints**

1. **Demand limitation**  
   \[
   x_i \;\le\; D_i \qquad \forall i\in I
   \]

2. **Inventory limitation**  
   \[
   x_i \;\le\; S_i \qquad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(R_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(D_i\) | Demand | `file_0_view_0` | `Demand` |
| \(S_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | Product Name filter | `file_0_view_0` | `Product Name` (contains “27in”) |

All parameter values are taken directly from the rows returned by the filter on *Product Name* containing “27in”. No additional data or constraints are introduced beyond those specified in the query.