#### Abstract Mathematical Model

**Index Sets:**
- $P$: set of products (from `factory_products_100.csv`, column `product`)
- $R$: set of resources (from `resources_capacities.csv`, column `resource`)

**Parameters:**
- $b$: batch size in units (from `factory_products_100.csv`, column `batch_size_units`; constant across all products)
- $p_i$: profit per unit of product $i \in P$ (from `factory_products_100.csv`, column `profit_per_unit`)
- $a_{ir}$: units of resource $r \in R$ consumed per unit of product $i \in P$ (from `factory_products_100.csv`, columns `r1_per_unit`, `r2_per_unit`, `r3_per_unit`)
- $d_i$: upper demand (units) for product $i \in P$ (from `factory_products_100.csv`, column `upper_demand_units`)
- $C_r$: total available amount of resource $r \in R$ (from `resources_capacities.csv`, column `capacity`)

**Decision Variables:**
- $x_i \in \mathbb{Z}_+$: number of batches of product $i \in P$ to produce (integer, $x_i \geq 0$)

**Objective:**
\[
\max \sum_{i \in P} b \cdot x_i \cdot p_i
\]

**Constraints:**

1. **Resource Capacity Constraints:**
   \[
   \sum_{i \in P} b \cdot a_{ir} \cdot x_i \leq C_r, \quad \forall r \in R
   \]

2. **Demand Upper Bound Constraints:**
   \[
   b \cdot x_i \leq d_i, \quad \forall i \in P
   \]

3. **Batch Integer Constraints:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in P
   \]

---

#### Data Mapping

- **factory_products_100.csv**
  - `product` $\rightarrow$ $P$
  - `batch_size_units` $\rightarrow$ $b$
  - `profit_per_unit` $\rightarrow$ $p_i$
  - `r1_per_unit`, `r2_per_unit`, `r3_per_unit` $\rightarrow$ $a_{ir}$ (for $r$ = R1, R2, R3)
  - `upper_demand_units` $\rightarrow$ $d_i$
- **resources_capacities.csv**
  - `resource` $\rightarrow$ $R$
  - `capacity` $\rightarrow$ $C_r$

All data is used as returned by CSVQA, with no additional filtering.