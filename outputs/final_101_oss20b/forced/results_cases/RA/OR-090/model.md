**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I}\; \text{batch\_size\_units}_i \; x_i \; \text{profit\_per\_unit}_i \\[4pt]
\text{subject to}\quad
& \sum_{i\in I}\; \text{batch\_size\_units}_i \; x_i \; \text{r1\_per\_unit}_i \;\le\; \text{capacity}_{R1} \\
& \sum_{i\in I}\; \text{batch\_size\_units}_i \; x_i \; \text{r2\_per\_unit}_i \;\le\; \text{capacity}_{R2} \\
& \sum_{i\in I}\; \text{batch\_size\_units}_i \; x_i \; \text{r3\_per\_unit}_i \;\le\; \text{capacity}_{R3} \\[4pt]
& \text{batch\_size\_units}_i \; x_i \;\le\; \text{upper\_demand\_units}_i \qquad \forall i\in I \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0} \qquad \forall i\in I
\end{aligned}
\]

*Indices*  
\(I\) – set of products (P1–P100)  
\(R\) – set of resources \(\{R1,R2,R3\}\)

*Parameters* – all taken directly from the CSV files (see Data Mapping below)

---

### Data Mapping

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| `product` | `file_0_view_0` | `product` |
| `profit_per_unit` | `file_0_view_0` | `profit_per_unit` |
| `r1_per_unit` | `file_0_view_0` | `r1_per_unit` |
| `r2_per_unit` | `file_0_view_0` | `r2_per_unit` |
| `r3_per_unit` | `file_0_view_0` | `r3_per_unit` |
| `upper_demand_units` | `file_0_view_0` | `upper_demand_units` |
| `batch_size_units` | `file_0_view_0` | `batch_size_units` |
| `resource` | `file_1_view_0` | `resource` |
| `capacity` | `file_1_view_0` | `capacity` |

*Index Sets*  
- \(I\) is the set of all `product` values in `file_0_view_0`.  
- \(R\) is the set of all `resource` values in `file_1_view_0`.

All numerical coefficients are taken as real numbers from the corresponding columns; the decision variables \(x_i\) are non‑negative integers. No additional data are invented or omitted.