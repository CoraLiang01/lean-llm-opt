#### Symbolic Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products where the value in column "Product Name" begins with 'S700_'.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: revenue per unit of product $i$ (from column "Revenue")
- $d_i$: total demand for product $i$ (from column "Demand")
- $I_i$: initial inventory for product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i \in \mathbb{Z}_+$: number of units of product $i$ to fulfill (non-negative integer)

**Objective:**  
\[
\max \sum_{i \in \mathcal{I}} A_i x_i
\]

**Constraints:**
1. **Inventory constraint:**  
   \[
   x_i \leq I_i \quad \forall i \in \mathcal{I}
   \]
2. **Demand constraint:**  
   \[
   x_i \leq d_i \quad \forall i \in \mathcal{I}
   \]
3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

---

#### Data Mapping

- **Index Set $\mathcal{I}$:**  
  All records in table_id `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv` where column "Product Name" starts with 'S700_'.

- **Parameter $A_i$:**  
  Column "Revenue" in table `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv`.

- **Parameter $d_i$:**  
  Column "Demand" in table `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv`.

- **Parameter $I_i$:**  
  Column "Initial Inventory" in table `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv`.