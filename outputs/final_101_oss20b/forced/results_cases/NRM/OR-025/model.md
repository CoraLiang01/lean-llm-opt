**Symbolic Model**

Let  

- \(I\) – set of all tablet models that satisfy the filter “Product Name contains ‘TABLET’”.  
- For each \(i\in I\)  

  \[
  \begin{aligned}
  R_i &\;=\;\text{Revenue of model }i \quad (\text{from column “Revenue”})\\
  D_i &\;=\;\text{Demand of model }i \quad (\text{from column “Demand”})\\
  S_i &\;=\;\text{Initial inventory of model }i \quad (\text{from column “Initial Inventory”})
  \end{aligned}
  \]

Decision variable  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad\text{(units of model }i\text{ to be sold)}
\]

---

**Objective**

\[
\max \;\sum_{i\in I} R_i\,x_i
\]

---

**Constraints**

1. **Demand limits**  
   \[
   x_i \;\le\; D_i \qquad \forall i\in I
   \]

2. **Inventory limits**  
   \[
   x_i \;\le\; S_i \qquad \forall i\in I
   \]

3. **Non‑negativity / integrality**  
   \[
   x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
   \]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(R_i\) | Revenue | `file_0_view_0` | `Revenue` |
| \(D_i\) | Demand | `file_0_view_0` | `Demand` |
| \(S_i\) | Initial Inventory | `file_0_view_0` | `Initial Inventory` |
| \(I\) | Set of rows where `Product Name` contains “TABLET” | `file_0_view_0` | `Product Name` |

All parameter values are taken directly from the returned rows of the filtered view `file_0_view_0`. No additional data or constraints are introduced beyond those specified in the query.