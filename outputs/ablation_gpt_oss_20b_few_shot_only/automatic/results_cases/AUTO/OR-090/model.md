**Mathematical Model**

Let  

- \(P\) be the set of products \(\{P_1,\dots ,P_{100}\}\).  
- \(R=\{R1,R2,R3\}\) be the set of resources.

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(p_i\) | profit per unit of product \(i\) | `factory_products_100.csv` – column `profit_per_unit` |
| \(r_{1i}\) | consumption of resource R1 per unit of product \(i\) | `factory_products_100.csv` – column `r1_per_unit` |
| \(r_{2i}\) | consumption of resource R2 per unit of product \(i\) | `factory_products_100.csv` – column `r2_per_unit` |
| \(r_{3i}\) | consumption of resource R3 per unit of product \(i\) | `factory_products_100.csv` – column `r3_per_unit` |
| \(d_i\) | upper demand (units) for product \(i\) | `factory_products_100.csv` – column `upper_demand_units` |
| \(b_i\) | batch size (units per batch) for product \(i\) | `factory_products_100.csv` – column `batch_size_units` |
| \(c_R\) | total capacity of resource \(R\) | `resources_capacities.csv` – column `capacity` |

**Decision Variables**

\[
x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in P
\]

\(x_i\) = number of batches of product \(i\) produced.

**Objective**

\[
\max \;\sum_{i\in P} 10\,p_i\,x_i
\]

(10 is the common batch size; it can be replaced by \(b_i\) if desired.)

**Constraints**

1. **Resource capacities**

\[
\sum_{i\in P} 10\,r_{1i}\,x_i \;\le\; c_{R1}\qquad
\sum_{i\in P} 10\,r_{2i}\,x_i \;\le\; c_{R2}\qquad
\sum_{i\in P} 10\,r_{3i}\,x_i \;\le\; c_{R3}
\]

2. **Demand limits**

\[
10\,x_i \;\le\; d_i \qquad \forall i\in P
\]

3. **Non‑negativity & integrality**

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in P
\]

---

### Data Mapping

| Parameter | File Path | Column Name |
|-----------|-----------|-------------|
| \(p_i\) | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv` | `profit_per_unit` |
| \(r_{1i}\) | same file | `r1_per_unit` |
| \(r_{2i}\) | same file | `r2_per_unit` |
| \(r_{3i}\) | same file | `r3_per_unit` |
| \(d_i\) | same file | `upper_demand_units` |
| \(b_i\) | same file | `batch_size_units` |
| \(c_{R1}\) | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv` | `capacity` (row where `resource` = `R1`) |
| \(c_{R2}\) | same file | `capacity` (row where `resource` = `R2`) |
| \(c_{R3}\) | same file | `capacity` (row where `resource` = `R3`) |

All other data are derived from these mappings; no additional numeric values are introduced.