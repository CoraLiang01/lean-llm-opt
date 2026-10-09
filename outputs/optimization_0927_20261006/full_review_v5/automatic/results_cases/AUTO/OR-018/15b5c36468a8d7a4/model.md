#### Index Sets

- $I$: Set of all products classified under ‘Baby’ (from column "Product Name" with prefix "Baby" in table_id file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit of product $i \in I$ (from column "Revenue", table_id file_0_view_0).
- $d_i$: Demand for product $i \in I$ (from column "Demand", table_id file_0_view_0).
- $I_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory", table_id file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory and Demand Fulfillment Bounds:
   $$
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   $$

2. Integrality:
   $$
   x_i \in \mathbb{Z}, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: file_0_view_0 (from Salesdata.csv)
- Columns used:
    - Product Name (filtered: prefix "Baby")
    - Revenue
    - Demand
    - Initial Inventory

All parameters and index sets are defined using the exact columns and filter as returned by CSVQA. No additional constraints or data sources are used.