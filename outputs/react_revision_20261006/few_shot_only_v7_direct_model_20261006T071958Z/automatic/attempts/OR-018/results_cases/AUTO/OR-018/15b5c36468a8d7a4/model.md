##### Abstract Mathematical Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all products in the current data whose "Product Name" contains the substring "Baby".

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column "Revenue")
- $d_i$: Demand for product $i$ (from column "Demand")
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
$\displaystyle \max \sum_{i \in \mathcal{I}} A_i x_i$

**Constraints:**
- Demand fulfillment: $\quad x_i \leq d_i \quad \forall i \in \mathcal{I}$
- Inventory limit: $\quad x_i \leq I_i \quad \forall i \in \mathcal{I}$
- Nonnegativity and integrality: $\quad x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

---

**Data Mapping:**  
- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv`, table_id: `file_0_view_0`
- Index set $\mathcal{I}$: All records where `Product Name` contains "Baby"
- $A_i$: column `Revenue`
- $d_i$: column `Demand`
- $I_i$: column `Initial Inventory`