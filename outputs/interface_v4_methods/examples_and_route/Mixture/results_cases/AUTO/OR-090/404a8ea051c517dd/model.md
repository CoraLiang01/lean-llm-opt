## Abstract Mathematical Model

### Index Sets
- $I$: set of products (indexed by $i$), from `file_0_view_0.product`
- $K$: set of resources (indexed by $k$), from `file_1_view_0.resource$

### Parameters
- $b$: batch size in units (common to all products), from `file_0_view_0.batch_size_units$
- $p_i$: profit per unit of product $i$, from `file_0_view_0.profit_per_unit$
- $a_{ik}$: units of resource $k$ consumed per unit of product $i$, from `file_0_view_0.r1_per_unit`, `file_0_view_0.r2_per_unit`, `file_0_view_0.r3_per_unit$
- $d_i$: upper demand (units) for product $i$, from `file_0_view_0.upper_demand_units$
- $C_k$: total available amount of resource $k$, from `file_1_view_0.capacity$

### Decision Variables
- $x_i \in \mathbb{Z}_{\geq 0}$: number of batches of product $i$ to produce/purchase

### Objective
\[
\max \sum_{i \in I} b \cdot p_i \cdot x_i
\]

### Constraints

#### 1. Resource Capacity Constraints
\[
\sum_{i \in I} a_{ik} \cdot b \cdot x_i \leq C_k \qquad \forall k \in K
\]

#### 2. Demand Upper Bound Constraints
\[
b \cdot x_i \leq d_i \qquad \forall i \in I
\]

#### 3. Integrality and Nonnegativity
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

---

## Data Mapping

- $I$: `file_0_view_0.product`
- $K$: `file_1_view_0.resource`
- $b$: `file_0_view_0.batch_size_units` (all rows identical, use any row)
- $p_i$: `file_0_view_0.profit_per_unit` (keyed by `product`)
- $a_{i,\text{R1}}$: `file_0_view_0.r1_per_unit` (keyed by `product`)
- $a_{i,\text{R2}}$: `file_0_view_0.r2_per_unit` (keyed by `product`)
- $a_{i,\text{R3}}$: `file_0_view_0.r3_per_unit` (keyed by `product`)
- $d_i$: `file_0_view_0.upper_demand_units` (keyed by `product`)
- $C_k$: `file_1_view_0.capacity` (keyed by `resource`)

---

All parameters and constraints are mapped directly to the supplied data columns and identifiers. No data or constraints are omitted.