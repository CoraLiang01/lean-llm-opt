#### Abstract Mathematical Model

Let $I$ be the set of products, indexed by $i$.

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from RetailSalesDataset.csv, column 'Revenue')
- $d_i$: Demand for product $i$ (from RetailSalesDataset.csv, column 'Demand')
- $s_i$: Initial Inventory for product $i$ (from RetailSalesDataset.csv, column 'Initial Inventory')

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill (allocate), $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. Inventory limit for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand limit for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Nonnegativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$: All products in RetailSalesDataset.csv, column 'Product Name', table_id: file_0_view_0
- $r_i$: RetailSalesDataset.csv, column 'Revenue', table_id: file_0_view_0, key: 'Product Name'
- $d_i$: RetailSalesDataset.csv, column 'Demand', table_id: file_0_view_0, key: 'Product Name'
- $s_i$: RetailSalesDataset.csv, column 'Initial Inventory', table_id: file_0_view_0, key: 'Product Name'

Each constraint and parameter is mapped directly to the corresponding column and product identifier in the source data. No data is omitted or synthesized.