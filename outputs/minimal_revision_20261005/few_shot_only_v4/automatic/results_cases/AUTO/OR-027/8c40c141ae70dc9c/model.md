#### Index Sets

- Let $I$ be the set of all products in the "Sub Category" column classified under 'Organ'.

#### Parameters

- $A_i$: Revenue per unit of product $i$, from column "Revenue".
- $d_i$: Demand for product $i$, from column "Demand".
- $I_i$: Initial inventory of product $i$, from column "Initial Inventory".

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint**  
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraint**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Variable Domain**  
   $$
   x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$**: All records in table_id `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv` where `"Sub Category"` is classified under 'Organ'.
- **Parameter $A_i$**: `"Revenue"` column of the above table.
- **Parameter $d_i$**: `"Demand"` column of the above table.
- **Parameter $I_i$**: `"Initial Inventory"` column of the above table.