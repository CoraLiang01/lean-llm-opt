#### Index Sets

- $I$: Set of all products (indexed by $i$), where each product is identified by its "Product Name" in the data.

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column "Revenue").
- $d_i$: Total demand for product $i$ (from column "Demand").
- $I_i$: Initial inventory available for product $i$ (from column "Initial Inventory").

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Demand fulfillment constraint:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]

2. **Inventory constraint:**
   \[
   x_i \leq I_i \quad \forall i \in I
   \]

3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index set $I$:** All unique values in column "Product Name" of table_id `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv`
- **Parameter $A_i$:** Column "Revenue" of the same table
- **Parameter $d_i$:** Column "Demand" of the same table
- **Parameter $I_i$:** Column "Initial Inventory" of the same table