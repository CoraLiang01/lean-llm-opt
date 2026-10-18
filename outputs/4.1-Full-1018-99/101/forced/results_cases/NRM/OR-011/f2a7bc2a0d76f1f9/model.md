#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of products classified under ‘id999’.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: Demand for product $i \in I$ during the sales horizon (from column ‘Demand’).
- $I_i$: Initial inventory of product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** OnlineRetailSalesDataset.csv
- **Index Set $I$:** All records where column ‘id_number’ = ‘id999’ (predicate: id_number = ‘id999’).
- **Parameter $A_i$:** Column ‘Revenue’ in OnlineRetailSalesDataset.csv.
- **Parameter $d_i$:** Column ‘Demand’ in OnlineRetailSalesDataset.csv.
- **Parameter $I_i$:** Column ‘Initial Inventory’ in OnlineRetailSalesDataset.csv.