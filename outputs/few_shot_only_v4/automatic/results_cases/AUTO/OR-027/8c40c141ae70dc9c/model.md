#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : Set of all products classified under ‘Organ’.

**Parameters:**

- $A_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv).
- $d_i$ : Demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraints:**
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraints:**
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Variable Domain:**
   $$
   x_i \in \mathbb{Z}_{+}, \quad \forall i \in I
   $$

---

#### Data Mapping

- **File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv
- **Index Set $I$:** All rows where the product is classified under ‘Organ’ (as defined by the user or data schema).
- **Parameter $A_i$:** Column ‘Revenue’.
- **Parameter $d_i$:** Column ‘Demand’.
- **Parameter $I_i$:** Column ‘Initial Inventory’.