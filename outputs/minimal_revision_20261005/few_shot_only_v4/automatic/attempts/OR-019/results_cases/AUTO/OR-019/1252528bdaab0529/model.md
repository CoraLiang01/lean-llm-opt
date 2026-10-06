#### Index Sets

- $I$: Set of all products with "27in" in the "Product Name" column.

#### Parameters

- $A_i$: Revenue per unit of product $i \in I$ (from "Revenue").
- $d_i$: Demand for product $i \in I$ (from "Demand").
- $I_i$: Initial inventory for product $i \in I$ (from "Initial Inventory").

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Demand fulfillment:** $x_i \leq d_i \quad \forall i \in I$
2. **Inventory limit:** $x_i \leq I_i \quad \forall i \in I$
3. **Nonnegativity and integrality:** $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv`
    - Index set $I$: All rows where `"Product Name"` contains `"27in"`
    - Parameter $A_i$: `"Revenue"` column
    - Parameter $d_i$: `"Demand"` column
    - Parameter $I_i$: `"Initial Inventory"` column