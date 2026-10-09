---

### Abstract Mathematical Model

#### Index Sets
- Let $\mathcal{I}$ be the set of all products classified under ‘Aalop’.

#### Parameters
- $A_i$: Revenue per unit of product $i \in \mathcal{I}$ (from column "Revenue").
- $d_i$: Demand for product $i \in \mathcal{I}$ (from column "Demand").
- $I_i$: Initial inventory for product $i \in \mathcal{I}$ (from column "Initial Inventory").

#### Decision Variables
- $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$  
  (Number of units of product $i$ to fulfill; non-negative integer)

#### Objective
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

#### Constraints
1. **Inventory Constraint:**  
   \[
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

---

### Data Mapping

- **Source Table:**  
  - `file_0_view_0` (from "RestaurantSalesreport.csv")
- **Columns Used:**  
  - "Product Name" (for classification and indexing)
  - "Revenue" (parameter $A_i$)
  - "Demand" (parameter $d_i$)
  - "Initial Inventory" (parameter $I_i$)
- **Selection Rule:**  
  - $\mathcal{I}$ is the set of all records in `file_0_view_0` where "Product Name" is classified under ‘Aalop’. (If the classification is by substring, select all rows where "Product Name" contains ‘Aalop’.)

---