Mathematical Model

Index Sets:
Let $I$ be the set of all clothing products, indexed by $i$.

Parameters:
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory")

Decision Variables:
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

Objective:
$\max \sum_{i \in I} A_i x_i$

Constraints:
1. Inventory constraint: $x_i \leq I_i \quad \forall i \in I$
2. Demand constraint: $x_i \leq d_i \quad \forall i \in I$
3. Nonnegativity and integrality: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

Data Mapping:
- Table: file_0_view_0 (Salesofsummerclothes.csv)
- Index set $I$: All unique values in column "Product Name"
- Parameter $A_i$: column "Revenue"
- Parameter $d_i$: column "Demand"
- Parameter $I_i$: column "Initial Inventory"