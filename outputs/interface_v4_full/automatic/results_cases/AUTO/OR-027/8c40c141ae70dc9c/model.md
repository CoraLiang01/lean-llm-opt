#### Abstract Mathematical Model

Let:

- $I$ = index set of all products classified as ‘Organ’ (from Sub Category column)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter, from Revenue column)
    - $d_i$ = total demand for product $i$ (parameter, from Demand column)
    - $s_i$ = initial inventory of product $i$ (parameter, from Initial Inventory column)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Variables:**

- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. Inventory constraint:
   $$
   x_i \leq s_i, \quad \forall i \in I
   $$
2. Demand constraint:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table ID: file_0_view_0 (from SupermartGrocerySales-RetailAnalyticsDataset.csv)
    - Product identifier: Sub Category
    - Revenue parameter: Revenue
    - Demand parameter: Demand
    - Initial inventory parameter: Initial Inventory

All parameters and index sets are defined symbolically; no literal values or record counts are included.