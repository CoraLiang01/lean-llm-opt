#### Index Sets

- $I$: Set of all ‘TABLET’ smartphone models, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit for model $i$.  
  [From: file_0_view_0, column "Revenue"]

- $d_i$: Total demand for model $i$ over the sales horizon.  
  [From: file_0_view_0, column "Demand"]

- $s_i$: Initial inventory for model $i$.  
  [From: file_0_view_0, column "Initial Inventory"]

#### Decision Variables

- $x_i$: Number of units of model $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \, x_i
$$

#### Constraints

1. **Inventory Constraint:**  
  $\forall i \in I: \quad x_i \leq s_i$

2. **Demand Constraint:**  
  $\forall i \in I: \quad x_i \leq d_i$

3. **Non-negativity and Integrality:**  
  $\forall i \in I: \quad x_i \in \mathbb{Z}_+, \ x_i \geq 0$

---

#### Data Mapping

- Table: file_0_view_0 (from SmartphoneRetailOutletSalesData.csv)
    - Index set $I$: All rows where "Product Name" starts with "TABLET"
    - $A_i$: "Revenue"
    - $d_i$: "Demand"
    - $s_i$: "Initial Inventory"