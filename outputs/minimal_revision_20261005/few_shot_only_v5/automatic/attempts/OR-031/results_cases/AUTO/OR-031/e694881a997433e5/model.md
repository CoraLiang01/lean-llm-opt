#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all dairy products, indexed by $i$.

**Parameters:**  
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")  
- $d_i$: Demand for product $i$ (from column "Demand")  
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")  

**Decision Variables:**  
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$  
  Domain: $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**  
1. **Demand fulfillment:**  
  $x_i \leq d_i, \quad \forall i \in I$

2. **Inventory limit:**  
  $x_i \leq I_i, \quad \forall i \in I$

3. **Non-negativity and integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Index Set $I$:**  
  All records in table_id: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv`, column: `Full_Product_Name`

- **Parameter $A_i$:**  
  table_id: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv`, column: `Revenue`

- **Parameter $d_i$:**  
  table_id: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv`, column: `Demand`

- **Parameter $I_i$:**  
  table_id: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv`, column: `Initial Inventory`