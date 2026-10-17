#### Index Sets

- $I$: Set of products classified under ‘id999’.

#### Parameters

- $A_i$: Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory and Demand Fulfillment Bounds:
   $$
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   $$

   (Equivalently, two constraints per product:)
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

2. Integer Non-negativity:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: OnlineRetailSalesDataset.csv (table_id: file_0_view_0)
- Columns used:
    - id_number (filtered: exact match ‘id999’)
    - Revenue
    - Demand
    - Initial Inventory

All parameters and index sets are defined using the records returned by the filter id_number = ‘id999’ from the specified table and columns.