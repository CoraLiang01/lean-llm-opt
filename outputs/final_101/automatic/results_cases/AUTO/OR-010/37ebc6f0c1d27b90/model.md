#### Index Sets

- $I$: set of all products (indexed by $i$), corresponding to all "Product Name" entries in table_id file_0_view_0.

#### Parameters

- $A_i$: revenue per unit for product $i$ (from column "Revenue" in table_id file_0_view_0)
- $d_i$: deterministic demand for product $i$ during the sales cycle (from column "Demand" in table_id file_0_view_0)
- $I_i$: initial inventory for product $i$ (from column "Initial Inventory" in table_id file_0_view_0)

#### Decision Variables

- $x_i$: number of orders fulfilled for product $i$, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory constraints:
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. Demand constraints:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: file_0_view_0 (from MobileSalesDataset.csv)
    - Index set $I$: "Product Name"
    - Parameter $A_i$: "Revenue"
    - Parameter $d_i$: "Demand"
    - Parameter $I_i$: "Initial Inventory"