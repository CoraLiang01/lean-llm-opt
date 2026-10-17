**Mathematical Model**

Let  

* \(I=\{P1,\dots ,P100\}\) – set of products  
* \(R=\{R1,R2,R3\}\)      – set of resources  

Parameters (all read from the CSV files)

| Parameter | Definition | Source |
|-----------|------------|--------|
| \(profit_{i}\) | profit per unit of product \(i\) | `factory_products_100.csv` – column `profit_per_unit` |
| \(r_{k,i}\) | consumption of resource \(k\) per unit of product \(i\) | `factory_products_100.csv` – columns `r1_per_unit`, `r2_per_unit`, `r3_per_unit` |
| \(U_{i}\) | upper demand bound (units) for product \(i\) | `factory_products_100.csv` – column `upper_demand_units` |
| \(b_{i}\) | batch size (units per batch) for product \(i\) | `factory_products_100.csv` – column `batch_size_units` (all 10) |
| \(C_{k}\) | total available amount of resource \(k\) | `resources_capacities.csv` – column `capacity` |

Decision variables  

\[
x_{i}\in \mathbb{Z}_{+}\qquad \forall i\in I
\]

*\(x_{i}\)* = number of batches of product \(i\) produced.

---

### Objective

\[
\max \;\; \sum_{i\in I} b_{i}\,x_{i}\,profit_{i}
\]

---

### Constraints  

**Demand limits**

\[
b_{i}\,x_{i}\;\le\; U_{i}\qquad \forall i\in I
\]

**Resource limits**

\[
\sum_{i\in I} b_{i}\,x_{i}\,r_{k,i}\;\le\; C_{k}\qquad \forall k\in R
\]

**Non‑negativity & integrality**

\[
x_{i}\in \mathbb{Z}_{+}\qquad \forall i\in I
\]

---

**Data Mapping Rules**

| Table ID | Column | Mapped Parameter |
|----------|--------|------------------|
| `file_0_view_0` | `product` | index \(i\in I\) |
| | `profit_per_unit` | \(profit_{i}\) |
| | `r1_per_unit` | \(r_{1,i}\) |
| | `r2_per_unit` | \(r_{2,i}\) |
| | `r3_per_unit` | \(r_{3,i}\) |
| | `upper_demand_units` | \(U_{i}\) |
| | `batch_size_units` | \(b_{i}\) |
| `file_1_view_0` | `resource` | index \(k\in R\) |
| | `capacity` | \(C_{k}\) |

All indices and parameters are taken directly from the complete runtime records; no values are enumerated or omitted.