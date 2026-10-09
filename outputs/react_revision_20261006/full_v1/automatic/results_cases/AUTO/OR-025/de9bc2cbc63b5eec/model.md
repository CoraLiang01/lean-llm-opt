#### Index Sets

- $I$: Set of all ‘TABLET’ smartphone models, indexed by $i$.

#### Parameters

- $r_i$: Revenue per unit of model $i$.  
  (From column "Revenue" in table_id: file_0_view_0)
- $d_i$: Demand for model $i$.  
  (From column "Demand" in table_id: file_0_view_0)
- $s_i$: Initial inventory for model $i$.  
  (From column "Initial Inventory" in table_id: file_0_view_0)

#### Decision Variables

- $x_i$: Number of units of model $i$ to fulfill, $x_i \in \mathbb{Z}_+$, for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} r_i x_i
$$

#### Constraints

1. Inventory constraint for each model:
   $$
   x_i \leq s_i \quad \forall i \in I
   $$
2. Demand constraint for each model:
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Index set $I$ and all parameters $r_i$, $d_i$, $s_i$ are defined by all records in table_id: file_0_view_0, columns "Product Name", "Revenue", "Demand", "Initial Inventory" of SmartphoneRetailOutletSalesData.csv, filtered to products where "Product Name" has prefix "TABLET".