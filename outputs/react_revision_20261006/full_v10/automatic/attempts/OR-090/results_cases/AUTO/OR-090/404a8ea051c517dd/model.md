## Symbolic Mathematical Model

**Sets**
- $I$: set of products (from file_0_view_0, column product), $|I|=100$
- $K$: set of resources (from file_1_view_0, column resource), $K = \{\text{R1}, \text{R2}, \text{R3}\}$

**Parameters**
- $profit\_per\_unit_i$: profit per unit of product $i$ (file_0_view_0, column profit_per_unit)
- $r_{ik}$: units of resource $k$ consumed per unit of product $i$ (file_0_view_0, columns r1_per_unit, r2_per_unit, r3_per_unit)
- $demand\_ub_i$: upper demand (units) for product $i$ (file_0_view_0, column upper_demand_units)
- $batch\_size$: batch size in units (file_0_view_0, column batch_size_units, all equal to 10)
- $cap_k$: total available amount of resource $k$ (file_1_view_0, column capacity)

**Decision Variables**
- $x_i \in \mathbb{Z}_+$: number of batches of product $i$ to produce (integer, $x_i \geq 0$)

**Objective**
\[
\max \sum_{i \in I} batch\_size \cdot x_i \cdot profit\_per\_unit_i
\]

**Constraints**
- Resource capacities (for each $k \in K$):
\[
\sum_{i \in I} batch\_size \cdot x_i \cdot r_{ik} \leq cap_k
\]
- Demand upper bounds (for each $i \in I$):
\[
batch\_size \cdot x_i \leq demand\_ub_i
\]
- Integer batches:
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in I
\]

---

## Data Mapping

- $I$ (products): file_0_view_0, column product, all 100 rows
- $K$ (resources): file_1_view_0, column resource, all 3 rows
- $profit\_per\_unit_i$: file_0_view_0, column profit_per_unit, for each $i$
- $r_{ik}$: file_0_view_0, columns r1_per_unit, r2_per_unit, r3_per_unit, for each $i$ and $k$
- $demand\_ub_i$: file_0_view_0, column upper_demand_units, for each $i$
- $batch\_size$: file_0_view_0, column batch_size_units (all 10)
- $cap_k$: file_1_view_0, column capacity, for each $k$ (resource)

All sets, parameters, and constraints are mapped directly to the columns and rows as described above. No entities are omitted.