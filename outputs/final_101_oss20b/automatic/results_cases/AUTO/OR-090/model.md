## Mathematical Model

**Indices**  
\(I=\{P_1,\dots ,P_{100}\}\) products  
\(R=\{R_1,R_2,R_3\}\) resources  

**Parameters**  

| Symbol | Definition | Source |
|--------|------------|--------|
| \(profit_{i}\) | profit per unit of product \(i\) | `file_0_view_0`.`profit_per_unit` |
| \(r_{1,i}\) | R1 consumption per unit of product \(i\) | `file_0_view_0`.`r1_per_unit` |
| \(r_{2,i}\) | R2 consumption per unit of product \(i\) | `file_0_view_0`.`r2_per_unit` |
| \(r_{3,i}\) | R3 consumption per unit of product \(i\) | `file_0_view_0`.`r3_per_unit` |
| \(U_i\) | upper demand (units) for product \(i\) | `file_0_view_0`.`upper_demand_units` |
| \(s_i\) | batch size (units) for product \(i\) (constant 10) | `file_0_view_0`.`batch_size_units` |
| \(C_r\) | total capacity of resource \(r\) | `file_1_view_0`.`capacity` |

**Decision Variables**  

\(x_i \in \mathbb{Z}_{\ge 0}\) number of batches of product \(i\)  

**Objective**  

\[
\max \;\sum_{i\in I} s_i\,profit_{i}\,x_i
\]

**Constraints**

1. **Demand limits**  
   \[
   s_i\,x_i \;\le\; U_i \qquad \forall i\in I
   \]

2. **Resource capacities**  
   \[
   \sum_{i\in I} s_i\,r_{1,i}\,x_i \;\le\; C_{R_1}
   \]
   \[
   \sum_{i\in I} s_i\,r_{2,i}\,x_i \;\le\; C_{R_2}
   \]
   \[
   \sum_{i\in I} s_i\,r_{3,i}\,x_i \;\le\; C_{R_3}
   \]

3. **Non‑negativity & integrality**  
   \[
   x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
   \]

---

## Data Mapping Rules

| Index Set | Source | Key Column |
|-----------|--------|------------|
| \(I\) | `file_0_view_0` | `product` |
| \(R\) | `file_1_view_0` | `resource` |

| Parameter | Table | Column |
|-----------|-------|--------|
| \(profit_{i}\) | `file_0_view_0` | `profit_per_unit` |
| \(r_{1,i}\) | `file_0_view_0` | `r1_per_unit` |
| \(r_{2,i}\) | `file_0_view_0` | `r2_per_unit` |
| \(r_{3,i}\) | `file_0_view_0` | `r3_per_unit` |
| \(U_i\) | `file_0_view_0` | `upper_demand_units` |
| \(s_i\) | `file_0_view_0` | `batch_size_units` |
| \(C_{R}\) | `file_1_view_0` | `capacity` |

All mappings use the exact identifiers from the source tables; no additional filtering or aggregation is applied beyond the indices defined above. The model uses the full set of 100 products and 3 resources as provided.