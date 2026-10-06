**Abstract Mathematical Model**

**Index Sets**
- $I$: set of products (from `file_0_view_0`, column `product`)
- $K$: set of resources (from `file_1_view_0`, column `resource$)

**Parameters**
- $b$: batch size in units (from `file_0_view_0`, column `batch_size_units$; all rows have $b=10$)
- $p_i$: profit per unit of product $i$ (from `file_0_view_0`, column `profit_per_unit$)
- $a_{ik}$: units of resource $k$ consumed per unit of product $i$ (from `file_0_view_0`, columns `r1_per_unit`, `r2_per_unit`, `r3_per_unit$; $k$ matches resource name)
- $d_i$: upper demand (units) for product $i$ (from `file_0_view_0`, column `upper_demand_units$)
- $C_k$: total available amount of resource $k$ (from `file_1_view_0`, column `capacity$)

**Decision Variables**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of batches of product $i$ to produce/purchase

**Objective**
\[
\max \sum_{i \in I} b \cdot p_i \cdot x_i
\]

**Constraints**
1. **Resource Capacity Constraints** (for each $k \in K$):
   \[
   \sum_{i \in I} b \cdot a_{ik} \cdot x_i \leq C_k
   \]
2. **Demand Upper Bound Constraints** (for each $i \in I$):
   \[
   b \cdot x_i \leq d_i
   \]
3. **Batch Integrality** (for each $i \in I$):
   \[
   x_i \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $I$: All `product` in `file_0_view_0`
- $K$: All `resource` in `file_1_view_0`
- $b$: `batch_size_units` in `file_0_view_0` (all rows, value 10)
- $p_i$: `profit_per_unit` in `file_0_view_0`, matched by `product`
- $a_{ik}$: For $k=$R1, R2, R3, use `r1_per_unit`, `r2_per_unit`, `r3_per_unit` in `file_0_view_0`, matched by `product`
- $d_i$: `upper_demand_units` in `file_0_view_0`, matched by `product`
- $C_k$: `capacity` in `file_1_view_0`, matched by `resource$ (R1, R2, R3)

**Variable**
- $x_i$: Number of batches of product $i$ to produce/purchase (integer, nonnegative)

---

**Summary:**  
Choose integer batch quantities $x_i$ for each product $i$ to maximize total profit, subject to resource and demand limits, where each batch is $b=10$ units. All parameters and index sets are mapped directly from the supplied CSV data.