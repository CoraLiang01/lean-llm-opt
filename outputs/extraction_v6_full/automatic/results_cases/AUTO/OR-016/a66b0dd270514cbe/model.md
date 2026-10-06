#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all product categories, indexed by $i$ (e.g., electronics, apparel, homeware, etc.), with unique identifiers from the source.

**Parameters:**
- $A_i$: Revenue per unit for product $i$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All unique values in column `Product Name` from table_id `file_0_view_0` in `RetailSalesDataset.csv`.
- **Parameter $A_i$:** Value from column `Revenue` for each $i$ in table_id `file_0_view_0`.
- **Parameter $d_i$:** Value from column `Demand` for each $i$ in table_id `file_0_view_0`.
- **Parameter $I_i$:** Value from column `Initial Inventory` for each $i$ in table_id `file_0_view_0`.

(CSVQA_DATA bindings: 
- $A_i$: [Product Name], Revenue, file_0_view_0
- $d_i$: [Product Name], Demand, file_0_view_0
- $I_i$: [Product Name], Initial Inventory, file_0_view_0
)