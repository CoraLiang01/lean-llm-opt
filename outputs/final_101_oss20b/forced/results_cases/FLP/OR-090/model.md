**Symbolic Model**

Let  

* \(I=\{P1,\dots ,P100\}\) products  
* \(R=\{R1,R2,R3\}\) resources  

Parameters (all real, indexed by \(i\in I\) or \(r\in R\))

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(p_i\) | profit per unit of product \(i\) | `factory_products_100.csv` column `profit_per_unit` |
| \(c_{ri}\) | consumption of resource \(r\) per unit of product \(i\) | `factory_products_100.csv` columns `r1_per_unit`, `r2_per_unit`, `r3_per_unit` |
| \(u_i\) | upper demand bound (units) for product \(i\) | `factory_products_100.csv` column `upper_demand_units` |
| \(b_i\) | batch size (units) for product \(i\) | `factory_products_100.csv` column `batch_size_units` |
| \(C_r\) | total capacity of resource \(r\) | `resources_capacities.csv` column `capacity` |

Decision variables  

* \(x_i \in \mathbb{Z}_{\ge 0}\) number of batches of product \(i\)

---

**Objective**

\[
\max \;\; \sum_{i\in I} 10\,p_i\,x_i
\]

(10 is the batch size; it can be replaced by \(b_i\) if batch sizes differ.)

---

**Constraints**

1. **Resource limits**  
   \[
   \sum_{i\in I} 10\,c_{ri}\,x_i \;\le\; C_r , \qquad \forall r\in R
   \]

2. **Demand limits**  
   \[
   10\,x_i \;\le\; u_i , \qquad \forall i\in I
   \]

3. **Non‑negativity and integrality**  
   \[
   x_i \in \mathbb{Z}_{\ge 0}, \qquad \forall i\in I
   \]

---

**Data Mapping Rules**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(p_i\) | `file_0_view_0` | `profit_per_unit` |
| \(c_{R1,i}\) | `file_0_view_0` | `r1_per_unit` |
| \(c_{R2,i}\) | `file_0_view_0` | `r2_per_unit` |
| \(c_{R3,i}\) | `file_0_view_0` | `r3_per_unit` |
| \(u_i\) | `file_0_view_0` | `upper_demand_units` |
| \(b_i\) | `file_0_view_0` | `batch_size_units` |
| \(C_{R1}\) | `file_1_view_0` | `capacity` (row where `resource`=`R1`) |
| \(C_{R2}\) | `file_1_view_0` | `capacity` (row where `resource`=`R2`) |
| \(C_{R3}\) | `file_1_view_0` | `capacity` (row where `resource`=`R3`) |

All indices \(i\) and \(r\) are taken from the complete set of rows in the respective tables; no additional filtering is applied. The model uses the exact values from the CSV files as specified above.