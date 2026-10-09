#### Index Sets

- $I$: set of all products, indexed by $i$

#### Parameters

- $r_i$: revenue per unit of product $i$ (from column "Revenue")
- $d_i$: total demand for product $i$ over the sales horizon (from column "Demand")
- $s_i$: initial inventory of product $i$ (from column "Initial Inventory")

#### Decision Variables

- $x_i$: number of units of product $i$ to fulfill for customer purchases, $x_i \in \mathbb{Z}_+$

#### Objective

$$
\max \sum_{i \in I} r_i \cdot x_i
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

- Table: `file_0_view_0` (from `OnlineSalesDataset.csv`)
    - Index set $I$: all rows, using "Product Name" as the product identifier
    - Parameter $r_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"