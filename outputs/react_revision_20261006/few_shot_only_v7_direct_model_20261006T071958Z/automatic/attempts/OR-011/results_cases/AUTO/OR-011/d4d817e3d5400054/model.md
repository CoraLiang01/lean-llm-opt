#### Mathematical Model

**Index Sets:**
- $I$ : set of all products with identifier ‘id999’ in the dataset.

**Parameters:**
- $A_i$ : revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$ : deterministic demand for product $i \in I$ during the sales horizon (from column ‘Demand’).
- $I_i$ : initial inventory available for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$ : number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory and Demand Fulfillment:**
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   \]
   (Equivalently, two constraints per $i$: $x_i \leq I_i$, $x_i \leq d_i$.)

2. **Variable Domain:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All records in table_id `file_0_view_0` with `id_number = 'id999'`.
- **Parameter $A_i$:** `Revenue` column in table_id `file_0_view_0`.
- **Parameter $d_i$:** `Demand` column in table_id `file_0_view_0`.
- **Parameter $I_i$:** `Initial Inventory` column in table_id `file_0_view_0`.
- **Decision Variable $x_i$:** Fulfillment quantity for each $i \in I$.

All mappings are from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv`, table_id `file_0_view_0`, using the specified columns.