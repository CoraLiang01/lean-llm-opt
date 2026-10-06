#### Index Sets

- $I$: Set of all products (indexed by $i$).

#### Parameters

- $A_i$: Revenue per unit for product $i$ (from column "Revenue").
- $d_i$: Total demand for product $i$ over the sales cycle (from column "Demand").
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory").

#### Decision Variables

- $x_i$: Number of orders fulfilled for product $i$; $x_i \in \mathbb{Z}_+, \forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint** (cannot fulfill more than available inventory):
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraint** (cannot fulfill more than demand):
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Non-negativity and Integrality**:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv`
    - Product set $I$: column `"Product Name"`
    - Revenue $A_i$: column `"Revenue"`
    - Demand $d_i$: column `"Demand"`
    - Initial inventory $I_i$: column `"Initial Inventory"`