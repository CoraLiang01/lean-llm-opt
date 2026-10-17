**Symbolic Model**

Let  

* \(I=\{P_1,\dots ,P_{100}\}\) – set of products.  
* \(R=\{R_1,R_2,R_3\}\) – set of resources.  

Parameters (all real, indexed by \(i\in I\) or \(r\in R\))

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(p_i\) | profit per unit of product \(i\) | `file_0_view_0`.`profit_per_unit` |
| \(c_{ri}\) | consumption of resource \(r\) per unit of product \(i\) | `file_0_view_0`.`r1_per_unit`, `r2_per_unit`, `r3_per_unit` |
| \(d_i\) | upper demand bound (units) for product \(i\) | `file_0_view_0`.`upper_demand_units` |
| \(B\) | batch size (units) | fixed value 10 |
| \(C_r\) | total capacity of resource \(r\) | `file_1_view_0`.`capacity` |

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

( \(x_i\) = number of batches of product \(i\) produced.)

---

**Objective**

\[
\max \;\; \sum_{i\in I} B\,p_i\,x_i
\]

---

**Constraints**

1. **Resource limits**

\[
\sum_{i\in I} B\,c_{ri}\,x_i \;\le\; C_r \qquad \forall r\in R
\]

2. **Demand limits**

\[
B\,x_i \;\le\; d_i \qquad \forall i\in I
\]

3. **Non‑negativity & integrality**

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

---

**Data Mapping Rules**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(p_i\) | `file_0_view_0` | `profit_per_unit` |
| \(c_{1i}\) | `file_0_view_0` | `r1_per_unit` |
| \(c_{2i}\) | `file_0_view_0` | `r2_per_unit` |
| \(c_{3i}\) | `file_0_view_0` | `r3_per_unit` |
| \(d_i\) | `file_0_view_0` | `upper_demand_units` |
| \(C_{R1}\) | `file_1_view_0` | `capacity` (row where `resource` = `R1`) |
| \(C_{R2}\) | `file_1_view_0` | `capacity` (row where `resource` = `R2`) |
| \(C_{R3}\) | `file_1_view_0` | `capacity` (row where `resource` = `R3`) |

All other constants (batch size \(B=10\)) are fixed by the problem statement and not sourced from the CSV files.