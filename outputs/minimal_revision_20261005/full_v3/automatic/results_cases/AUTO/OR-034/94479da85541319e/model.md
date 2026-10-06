#### Sets
- $I$: Index set of all baked goods (from column "Product Name" in table_id: file_0_view_0).

#### Parameters
- $A_i$: Revenue per unit of baked good $i$ (from column "Revenue", table_id: file_0_view_0).
- $d_i$: Total demand for baked good $i$ (from column "Demand", table_id: file_0_view_0).
- $I_i$: Initial inventory for baked good $i$ (from column "Initial Inventory", table_id: file_0_view_0).

#### Decision Variables
- $x_i$: Quantity of baked good $i$ to fulfill, $\forall i \in I$.

#### Objective
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints
1. Inventory constraint:
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: file_0_view_0 (Frenchbakerydailysales.csv)
    - Set $I$: "Product Name"
    - Parameter $A_i$: "Revenue"
    - Parameter $d_i$: "Demand"
    - Parameter $I_i$: "Initial Inventory"