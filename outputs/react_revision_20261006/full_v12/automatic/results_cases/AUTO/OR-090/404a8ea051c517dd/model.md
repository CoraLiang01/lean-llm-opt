## Symbolic Mathematical Model

Let:
- $I$ = set of products, from all rows in file_0_view_0: $I = \{\text{P1}, \ldots, \text{P100}\}$
- $R$ = set of resources, from all rows in file_1_view_0: $R = \{\text{R1}, \text{R2}, \text{R3}\}$
- $x_i$ = integer number of batches of product $i \in I$ to produce (decision variable)
- $b$ = batch size in units (from any row in file_0_view_0, column batch_size_units): $b = 10$
- $p_i$ = profit per unit of product $i$ (file_0_view_0, profit_per_unit)
- $a_{ri}$ = resource $r$ consumed per unit of product $i$ (file_0_view_0, r1_per_unit, r2_per_unit, r3_per_unit)
- $d_i$ = upper demand (units) for product $i$ (file_0_view_0, upper_demand_units)
- $C_r$ = total available amount of resource $r$ (file_1_view_0, capacity)

**Objective:**
\[
\max \sum_{i \in I} b \cdot x_i \cdot p_i
\]

**Subject to:**

Resource constraints (for each $r \in R$):
\[
\sum_{i \in I} b \cdot x_i \cdot a_{ri} \leq C_r
\]

Demand upper bound (for each $i \in I$):
\[
b \cdot x_i \leq d_i
\]

Batch integrality (for each $i \in I$):
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

## Data Mapping

- $I$ (products): all rows in file_0_view_0, column "product"
- $R$ (resources): all rows in file_1_view_0, column "resource"
- $b$: file_0_view_0, column "batch_size_units" (constant, value 10)
- $p_i$: file_0_view_0, column "profit_per_unit"
- $a_{ri}$: file_0_view_0, columns "r1_per_unit", "r2_per_unit", "r3_per_unit" (for $r$ = R1, R2, R3)
- $d_i$: file_0_view_0, column "upper_demand_units"
- $C_r$: file_1_view_0, column "capacity" for each resource $r$

All indices, parameters, and constraints are mapped directly from the current CSV data as described.