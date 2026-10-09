##### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from all product values in file_0_view_0, column product)
- $K$ = set of resources, indexed by $k$ (from all resource values in file_1_view_0, column resource)
- $x_i$ = number of batches of product $i$ to produce or purchase (decision variable, integer, $x_i \geq 0$)

Parameters:
- $b_i$ = batch size in units for product $i$ (file_0_view_0, column batch_size_units)
- $p_i$ = profit per unit for product $i$ (file_0_view_0, column profit_per_unit)
- $a_{ik}$ = resource $k$ consumed per unit of product $i$ (file_0_view_0, columns r1_per_unit, r2_per_unit, r3_per_unit)
- $d_i$ = upper demand in units for product $i$ (file_0_view_0, column upper_demand_units)
- $C_k$ = total available amount of resource $k$ (file_1_view_0, column capacity)

Objective:
$$
\max \sum_{i \in I} b_i \cdot x_i \cdot p_i
$$

Subject to:
1. Resource capacity constraints (for each $k \in K$):
$$
\sum_{i \in I} b_i \cdot x_i \cdot a_{ik} \leq C_k
$$

2. Demand upper bound constraints (for each $i \in I$):
$$
b_i \cdot x_i \leq d_i
$$

3. Batch integrality and nonnegativity (for each $i \in I$):
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

##### Data Mapping

- $I$: All product values from file_0_view_0, column product
- $K$: All resource values from file_1_view_0, column resource
- $b_i$: file_0_view_0, column batch_size_units, keyed by product
- $p_i$: file_0_view_0, column profit_per_unit, keyed by product
- $a_{ik}$: file_0_view_0, columns r1_per_unit, r2_per_unit, r3_per_unit, keyed by product and resource (R1: r1_per_unit, R2: r2_per_unit, R3: r3_per_unit)
- $d_i$: file_0_view_0, column upper_demand_units, keyed by product
- $C_k$: file_1_view_0, column capacity, keyed by resource

- Decision variable $x_i$: number of batches of product $i$ to produce or purchase (integer, $\geq 0$)