## Symbolic Mathematical Model

**Sets**
- $I$: set of products, indexed by $i$ (from all product entries in file_0_view_0, i.e., $I = \{\text{P1}, \ldots, \text{P100}\}$)
- $R$: set of resources, indexed by $r$ (from all resource entries in file_1_view_0, i.e., $R = \{\text{R1}, \text{R2}, \text{R3}\}$)

**Parameters** (from Data Mapping)
- $b$: batch size in units (from any row in file_0_view_0, column batch_size_units; $b=10$)
- $p_i$: profit per unit of product $i$ (file_0_view_0, column profit_per_unit, row $i$)
- $a_{i,r}$: units of resource $r$ consumed per unit of product $i$ (file_0_view_0, columns r1_per_unit, r2_per_unit, r3_per_unit, row $i$)
- $d_i$: upper demand (units) for product $i$ (file_0_view_0, column upper_demand_units, row $i$)
- $C_r$: total available amount of resource $r$ (file_1_view_0, column capacity, row $r$)

**Decision Variables**
- $x_i \in \mathbb{Z}_+$: number of batches of product $i$ to produce (integer, $\geq 0$)

**Objective**
\[
\max \sum_{i \in I} b \cdot x_i \cdot p_i
\]

**Constraints**
1. **Resource constraints** (for each $r \in R$):
   \[
   \sum_{i \in I} b \cdot a_{i,r} \cdot x_i \leq C_r
   \]
2. **Demand upper bound** (for each $i \in I$):
   \[
   b \cdot x_i \leq d_i
   \]
3. **Batch integrality** (for each $i \in I$):
   \[
   x_i \in \mathbb{Z}_+, \quad x_i \geq 0
   \]

---

## Data Mapping

- $I$: All product entries in factory_products_100.csv (file_0_view_0, column product)
- $R$: All resource entries in resources_capacities.csv (file_1_view_0, column resource)
- $b$: batch_size_units (file_0_view_0, any row, column batch_size_units)
- $p_i$: profit_per_unit (file_0_view_0, column profit_per_unit, row $i$)
- $a_{i,\text{R1}}$: r1_per_unit (file_0_view_0, column r1_per_unit, row $i$)
- $a_{i,\text{R2}}$: r2_per_unit (file_0_view_0, column r2_per_unit, row $i$)
- $a_{i,\text{R3}}$: r3_per_unit (file_0_view_0, column r3_per_unit, row $i$)
- $d_i$: upper_demand_units (file_0_view_0, column upper_demand_units, row $i$)
- $C_r$: capacity (file_1_view_0, column capacity, row $r$)

**All indices, parameters, and constraints are mapped directly from the current CSV data as described above.**