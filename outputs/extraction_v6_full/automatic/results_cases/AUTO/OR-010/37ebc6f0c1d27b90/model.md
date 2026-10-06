#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : Set of all products, indexed by $i$ (from column “Product Name” in table_id file_0_view_0).

**Parameters:**

- $A_i$ : Revenue per unit for product $i$ (from column “Revenue” in table_id file_0_view_0).
- $d_i$ : Deterministic demand for product $i$ during the sales cycle (from column “Demand” in table_id file_0_view_0).
- $I_i$ : Initial inventory for product $i$ (from column “Initial Inventory” in table_id file_0_view_0).

**Decision Variables:**

- $x_i$ : Number of orders fulfilled for product $i$; $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**

\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**

1. **Inventory Constraint:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

2. **Demand Constraint:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$ (product index set): file_0_view_0, column “Product Name”
- $A_i$ (revenue): file_0_view_0, column “Revenue”
- $d_i$ (demand): file_0_view_0, column “Demand”
- $I_i$ (initial inventory): file_0_view_0, column “Initial Inventory”

All parameters are mapped using the CSVQA_DATA bindings validated for table_id file_0_view_0. No records or literal values are included in this abstract model.