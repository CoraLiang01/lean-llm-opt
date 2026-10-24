**Mathematical Model**

Let  

- \(I\) be the set of product identifiers whose `id_number` starts with **id999**.  
- For each \(i\in I\)  
  - \(Rev_i\)  – revenue per unit (from column **Revenue**)  
  - \(Dem_i\)  – known demand during the horizon (from column **Demand**)  
  - \(Inv_i\)  – available initial inventory (from column **Initial Inventory**)  

Decision variable  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) to fulfill.

\[
\begin{aligned}
\max\;& \sum_{i\in I} Rev_i \, x_i \\[4pt]
\text{s.t.}\;& x_i \;\le\; Inv_i && \forall i\in I \\[2pt]
& x_i \;\le\; Dem_i && \forall i\in I \\[2pt]
& x_i \;\ge\; 0 && \forall i\in I \\[2pt]
& x_i \in \mathbb{Z} && \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(I\) | `file_0_view_0` | `id_number` |
| \(Rev_i\) | `file_0_view_0` | `Revenue` |
| \(Dem_i\) | `file_0_view_0` | `Demand` |
| \(Inv_i\) | `file_0_view_0` | `Initial Inventory` |

All mappings use the exact identifiers and column names from the retrieved CSV data. No additional data or assumptions are introduced.