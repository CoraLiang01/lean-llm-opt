#### Index Sets

- $I$: Set of all products classified under ‘TABLET’ in table_id = file_0_view_0, column "Product Name".

#### Parameters

- $a_i$: Revenue per unit of product $i \in I$ (from column "Revenue", table_id = file_0_view_0).
- $d_i$: Demand for product $i \in I$ (from column "Demand", table_id = file_0_view_0).
- $s_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory", table_id = file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+$ (non-negative integers).

#### Objective

$$
\max \sum_{i \in I} a_i x_i
$$

#### Constraints

1. Inventory constraint for each product:
   $$
   x_i \leq s_i \qquad \forall i \in I
   $$
2. Demand constraint for each product:
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   $$

---

#### Data Mapping

- Table: file_0_view_0 (from "SmartphoneRetailOutletSalesData.csv")
    - Index set $I$: All rows where "Product Name" has prefix "TABLET"
    - $a_i$: "Revenue"
    - $d_i$: "Demand"
    - $s_i$: "Initial Inventory"