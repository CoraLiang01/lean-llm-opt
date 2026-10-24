**Symbolic Model**

Let  

- \(I\) – index set of all products whose *Product Name* starts with the prefix **S700_**.  
- For each \(i\in I\)  

  \[
  \begin{aligned}
  r_i &\;=\;\text{Revenue of product }i,\\
  d_i &\;=\;\text{Demand of product }i,\\
  I_i &\;=\;\text{Initial Inventory of product }i.
  \end{aligned}
  \]

Decision variable  

\[
x_i \;\in\; \mathbb{Z}_{+}\quad\text{(number of units of product }i\text{ to sell)}.
\]

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

1. **Demand limits**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limits**  
   \[
   x_i \;\le\; I_i \qquad \forall i\in I
   \]

3. **Non‑negativity / integrality**  
   \[
   x_i \;\in\; \mathbb{Z}_{+}\qquad \forall i\in I
   \]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(I_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | Product Name | `file_0_view_0` | `Product Name` (prefix `S700_`) |

All parameter values are taken directly from the rows of `file_0_view_0` that satisfy the prefix filter on *Product Name*. No additional data or constraints are introduced beyond those specified in the query.