#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products in the current data whose "Product Name" contains the substring "FAUX".

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**
1. **Inventory constraint:**  
   $x_i \leq I_i \quad \forall i \in I$
2. **Demand constraint:**  
   $x_i \leq d_i \quad \forall i \in I$
3. **Non-negativity and integrality:**  
   $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- **Source Table:**  
  `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv`  
  (table_id: `file_0_view_0`)

- **Index Set $I$:**  
  All records in `file_0_view_0` where the value in column `"Product Name"` contains the substring `"FAUX"`.

- **Parameter $A_i$:**  
  Value from column `"Revenue"` for product $i$ in `file_0_view_0`.

- **Parameter $d_i$:**  
  Value from column `"Demand"` for product $i$ in `file_0_view_0`.

- **Parameter $I_i$:**  
  Value from column `"Initial Inventory"` for product $i$ in `file_0_view_0`.