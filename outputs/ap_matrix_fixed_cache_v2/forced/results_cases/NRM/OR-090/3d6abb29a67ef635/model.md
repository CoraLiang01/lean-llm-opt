### Abstract Mathematical Model

#### Index Sets
- $P$: Set of products (from `factory_products_100.csv`, column `product`)
- $R$: Set of resources (from `resources_capacities.csv`, column `resource`)

#### Parameters
- $b$: Batch size in units (from `factory_products_100.csv`, column `batch_size_units$; constant for all products)
- $p_i$: Profit per unit of product $i \in P$ (from `factory_products_100.csv`, column `profit_per_unit`)
- $a_{i,r}$: Units of resource $r \in R$ consumed per unit of product $i \in P$ (from `factory_products_100.csv`, columns `r1_per_unit`, `r2_per_unit`, `r3_per_unit`)
- $d_i$: Maximum market demand (units) for product $i \in P$ (from `factory_products_100.csv`, column `upper_demand_units`)
- $C_r$: Total available amount of resource $r \in R$ (from `resources_capacities.csv`, column `capacity`)

#### Decision Variables
- $x_i \in \mathbb{Z}_+$: Number of batches of product $i \in P$ to produce (integer, $x_i \geq 0$)

#### Objective
\[
\max \sum_{i \in P} b \cdot x_i \cdot p_i
\]

#### Constraints

1. **Resource Capacity Constraints** (for each $r \in R$):
   \[
   \sum_{i \in P} b \cdot a_{i,r} \cdot x_i \leq C_r
   \]

2. **Demand Upper Bound Constraints** (for each $i \in P$):
   \[
   b \cdot x_i \leq d_i
   \]

3. **Batch Integer Constraints** (for each $i \in P$):
   \[
   x_i \in \mathbb{Z}_+, \quad x_i \geq 0
   \]

---

### Data Mapping

- **factory_products_100.csv**
  - Table ID: `file_0_view_0`
  - Columns used:
    - `product` $\rightarrow$ $P$
    - `profit_per_unit` $\rightarrow$ $p_i$
    - `r1_per_unit`, `r2_per_unit`, `r3_per_unit` $\rightarrow$ $a_{i,r}$ for $r \in \{\text{R1}, \text{R2}, \text{R3}\}$
    - `upper_demand_units` $\rightarrow$ $d_i$
    - `batch_size_units` $\rightarrow$ $b$ (constant for all $i$)
- **resources_capacities.csv**
  - Table ID: `file_1_view_0`
  - Columns used:
    - `resource` $\rightarrow$ $R$
    - `capacity` $\rightarrow$ $C_r$

All rows and columns from both tables are used as returned by the query. No additional filtering or aggregation is applied.