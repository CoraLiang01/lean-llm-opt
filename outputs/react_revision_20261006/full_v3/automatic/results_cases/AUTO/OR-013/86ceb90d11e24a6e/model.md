##### Mathematical Optimization Model

**Index Set:**
- $I$: set of all "4U" products (from the data, all products whose "Product Name" starts with "4U")

**Parameters:**
- $A_i$: revenue per unit of product $i$, from column "Revenue" in table_id file_0_view_0
- $d_i$: demand for product $i$, from column "Demand" in table_id file_0_view_0
- $I_i$: initial inventory for product $i$, from column "Initial Inventory" in table_id file_0_view_0

**Decision Variables:**
- $x_i$: number of units of product $i$ to fulfill, $\forall i \in I$; $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

##### Data Mapping

- Index set $I$: All records in table_id file_0_view_0 where "Product Name" starts with "4U"
- Parameter $A_i$: "Revenue" column in table_id file_0_view_0
- Parameter $d_i$: "Demand" column in table_id file_0_view_0
- Parameter $I_i$: "Initial Inventory" column in table_id file_0_view_0