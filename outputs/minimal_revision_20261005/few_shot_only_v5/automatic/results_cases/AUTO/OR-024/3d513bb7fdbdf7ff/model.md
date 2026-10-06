#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products with names beginning with "S700\_" from the table.

**Parameters:**  
For each $i \in I$:
- $a_i$: unit revenue of product $i$ (from column "Revenue")
- $d_i$: deterministic demand for product $i$ (from column "Demand")
- $s_i$: initial inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in I$:
- $x_i$: number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} a_i x_i
$$

**Constraints:**
1. **Inventory constraint:**  
  For all $i \in I$,
$$
x_i \leq s_i
$$

2. **Demand constraint:**  
  For all $i \in I$,
$$
x_i \leq d_i
$$

3. **Non-negativity and integrality:**  
  For all $i \in I$,
$$
x_i \in \mathbb{Z}_+, \quad x_i \geq 0
$$

---

#### Data Mapping

- **Index Set $I$:** All records in `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv` where `Product Name` starts with `"S700_"`.
- **Parameter $a_i$:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv`, column `Revenue`
- **Parameter $d_i$:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv`, column `Demand`
- **Parameter $s_i$:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv`, column `Initial Inventory`