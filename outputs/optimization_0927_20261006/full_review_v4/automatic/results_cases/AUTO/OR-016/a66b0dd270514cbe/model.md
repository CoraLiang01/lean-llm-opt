#### Index Sets

- $I$: Set of all products, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit for product $i$.  
- $d_i$: Demand for product $i$.  
- $I_i$: Initial Inventory for product $i$.

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. Inventory constraint for each product:
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
2. Demand constraint for each product:
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: file_0_view_0 (from RetailSalesDataset.csv)
    - Product Name: index set $I$
    - Revenue: parameter $A_i$
    - Demand: parameter $d_i$
    - Initial Inventory: parameter $I_i$
- All rows and columns from file_0_view_0 are used, as returned by CSVQA FALLBACK_FULL_DATA.