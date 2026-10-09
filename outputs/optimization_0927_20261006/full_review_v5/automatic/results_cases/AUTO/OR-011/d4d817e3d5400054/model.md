#### Abstract Mathematical Model

**Index Sets**

- $I$ : Set of products classified under ‘id999’.

**Parameters**

- $A_i$ : Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables**

- $x_i$ : Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints**

1. **Inventory Constraint:** 
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraint:** 
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:** 
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Source Table:** OnlineRetailSalesDataset.csv
- **Table ID:** file_0_view_0
- **Columns Used:** 
  - ‘id_number’ (filtered: exact match ‘id999’)
  - ‘Revenue’
  - ‘Demand’
  - ‘Initial Inventory’
- **Filter Applied:** Only rows where ‘id_number’ = ‘id999’ are included.