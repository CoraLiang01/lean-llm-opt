#### Index Sets

- $P$: set of products (from factory_products_100.csv, column product)
- $R$: set of resources (from resources_capacities.csv, column resource)

#### Parameters

- $b$: batch size in units (from factory_products_100.csv, column batch_size_units; constant across all products)
- $\pi_p$: profit per unit of product $p$ (from factory_products_100.csv, column profit_per_unit)
- $a_{pr}$: units of resource $r$ consumed per unit of product $p$ (from factory_products_100.csv, columns r1_per_unit, r2_per_unit, r3_per_unit)
- $d_p$: upper demand (units) for product $p$ (from factory_products_100.csv, column upper_demand_units)
- $C_r$: total available amount of resource $r$ (from resources_capacities.csv, column capacity)

#### Decision Variables

- $x_p \in \mathbb{Z}_+$: number of batches of product $p$ to produce/purchase

#### Objective

$$
\max \sum_{p \in P} b \cdot x_p \cdot \pi_p
$$

#### Constraints

1. **Resource Capacity Constraints** (for each $r \in R$):

$$
\sum_{p \in P} b \cdot a_{pr} \cdot x_p \leq C_r
$$

2. **Demand Upper Bound Constraints** (for each $p \in P$):

$$
b \cdot x_p \leq d_p
$$

3. **Batch Integer Constraints** (for each $p \in P$):

$$
x_p \in \mathbb{Z}_+, \quad x_p \geq 0
$$

#### Data Mapping

- factory_products_100.csv:
    - product $\rightarrow$ $P$
    - profit_per_unit $\rightarrow$ $\pi_p$
    - r1_per_unit, r2_per_unit, r3_per_unit $\rightarrow$ $a_{pr}$
    - upper_demand_units $\rightarrow$ $d_p$
    - batch_size_units $\rightarrow$ $b$
- resources_capacities.csv:
    - resource $\rightarrow$ $R$
    - capacity $\rightarrow$ $C_r$