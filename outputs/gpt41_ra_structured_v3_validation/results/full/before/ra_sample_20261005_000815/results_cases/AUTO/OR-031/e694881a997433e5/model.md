#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all dairy products, indexed by $i$.

**Parameters:**  
- $A_i$: Revenue per unit of product $i$ (from column "Revenue" in table_id: file_0_view_0)  
- $d_i$: Demand for product $i$ (from column "Demand" in table_id: file_0_view_0)  
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory" in table_id: file_0_view_0)

**Decision Variables:**  
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$  
  Domain: $x_i \in \mathbb{Z}_+, \; 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
$\max \sum_{i \in I} A_i \cdot x_i$

**Constraints:**  
1. Inventory and Demand Fulfillment:  
  $x_i \leq I_i, \quad \forall i \in I$  
  $x_i \leq d_i, \quad \forall i \in I$  
2. Non-negativity and Integrality:  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

**Data Mapping:**  
- Table: DairyGoodsSalesDataset.csv  
- table_id: file_0_view_0  
- Columns:  
  - Product identifier: "Full_Product_Name"  
  - Revenue: "Revenue"  
  - Demand: "Demand"  
  - Initial Inventory: "Initial Inventory"