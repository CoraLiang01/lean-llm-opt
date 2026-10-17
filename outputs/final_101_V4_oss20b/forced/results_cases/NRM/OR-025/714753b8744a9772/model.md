**Symbolic Model**

Let  

- \(I\) – set of all tablet models that satisfy the filter “Product Name contains ‘TABLET’”.  
- For each \(i\in I\)  

  \[
  \begin{aligned}
  r_i &\;=\;\text{Revenue of model }i,\\
  d_i &\;=\;\text{Demand of model }i,\\
  s_i &\;=\;\text{Initial Inventory of model }i.
  \end{aligned}
  \]

Decision variable  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

(fulfilment quantity of model \(i\)).

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

1. **Demand limits**  
   \[
   x_i \;\le\; d_i \qquad \forall i\in I
   \]

2. **Inventory limits**  
   \[
   x_i \;\le\; s_i \qquad \forall i\in I
   \]

3. **Non‑negativity / integrality**  
   \[
   x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
   \]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(r_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(d_i\) | Demand | `file_0_view_0` | `Demand` |
| \(s_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | Set of rows where `Product Name` contains “TABLET” | `file_0_view_0` | `Product Name` |

All parameter values are taken directly from the rows returned by the filter in the observation. No additional data or constraints are introduced.