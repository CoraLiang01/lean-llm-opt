#### Abstract Mathematical Optimization Model

**Index Set:**

- $I$ : set of products, indexed by $i$

**Parameters:**

- $r_i$ : revenue per unit of product $i$
- $d_i$ : total demand for product $i$ over the sales horizon
- $s_i$ : initial inventory of product $i$

**Decision Variables:**

- $x_i$ : number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

**Objective:**

\[
\max \sum_{i \in I} r_i \cdot x_i
\]

**Constraints:**

1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

2. **Demand fulfillment cannot exceed initial inventory:**
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]

3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$ : All unique values in column `Product Name` from table_id `file_0_view_0`
- $r_i$ : Value in column `Revenue` for product $i$ from table_id `file_0_view_0`
- $d_i$ : Value in column `Demand` for product $i$ from table_id `file_0_view_0`
- $s_i$ : Value in column `Initial Inventory` for product $i$ from table_id `file_0_view_0`

All data columns are sourced from table_id `file_0_view_0` (`WomenClothingEcommerceSalesData.csv`) as validated by CSVQA_DATA bindings.