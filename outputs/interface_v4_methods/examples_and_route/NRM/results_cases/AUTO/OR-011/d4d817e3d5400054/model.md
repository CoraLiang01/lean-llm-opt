#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of products with classification ‘id999’. (From OnlineRetailSalesDataset.csv, all rows where id_number = 'id999')

**Parameters:**
- $r_i$: Revenue per unit of product $i$.  
  Data Mapping: OnlineRetailSalesDataset.csv, column 'Revenue', table_id: file_0_view_0
- $d_i$: Demand for product $i$ during the sales horizon.  
  Data Mapping: OnlineRetailSalesDataset.csv, column 'Demand', table_id: file_0_view_0
- $s_i$: Initial inventory of product $i$.  
  Data Mapping: OnlineRetailSalesDataset.csv, column 'Initial Inventory', table_id: file_0_view_0

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill.  
  Domain: $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   $\quad x_i \leq s_i \quad \forall i \in I$

2. **Demand Constraint:**  
   $\quad x_i \leq d_i \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
   $\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

#### Data Mapping

- $I$: All rows in OnlineRetailSalesDataset.csv where id_number = 'id999' (table_id: file_0_view_0)
- $r_i$: OnlineRetailSalesDataset.csv, column 'Revenue', table_id: file_0_view_0
- $d_i$: OnlineRetailSalesDataset.csv, column 'Demand', table_id: file_0_view_0
- $s_i$: OnlineRetailSalesDataset.csv, column 'Initial Inventory', table_id: file_0_view_0

Each parameter and variable is indexed by the explicit business identifier 'id_number' as present in the data. No other products or columns are included.