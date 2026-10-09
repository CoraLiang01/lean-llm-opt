#### Symbolic Model

**Index Set:**  
Let $I$ be the set of all products in the dataset.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ during the sales cycle (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of orders fulfilled for product $i$  
  Domain: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**  
$\max \sum_{i \in I} A_i x_i$

**Constraints:**
1. **Inventory Constraint:**  
   $x_i \leq I_i, \quad \forall i \in I$
2. **Demand Constraint:**  
   $x_i \leq d_i, \quad \forall i \in I$
3. **Non-negativity and Integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Index Set $I$:**  
  All records in table_id: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv`, column: `"Product Name"`

- **Parameter $A_i$:**  
  table_id: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv`, column: `"Revenue"`

- **Parameter $d_i$:**  
  table_id: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv`, column: `"Demand"`

- **Parameter $I_i$:**  
  table_id: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv`, column: `"Initial Inventory"`