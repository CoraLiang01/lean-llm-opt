**Sets:**  
- $P$: Set of all products, indexed by $i$  
  (All records in `file_0_view_0`, column `Product Name`)

**Parameters:**  
- $A_i$: Revenue per unit for product $i$ (`file_0_view_0`, column `Revenue`)
- $d_i$: Demand for product $i$ (`file_0_view_0`, column `Demand`)
- $I_i$: Initial inventory for product $i$ (`file_0_view_0`, column `Initial Inventory`)

**Decision Variables:**  
- $x_i \in \mathbb{Z}_+, \quad \forall i \in P$  
  (Number of units of product $i$ to fulfill; non-negative integer)

**Objective:**  
\[
\max \sum_{i \in P} A_i \cdot x_i
\]

**Constraints:**  
1. **Inventory and Demand Bounds:**  
   \[
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in P
   \]

**Data Mapping:**  
- All sets and parameters are mapped from `file_0_view_0` in `RetailSalesDataset.csv`:
    - $P$: All rows, `Product Name`
    - $A_i$: `Revenue`
    - $d_i$: `Demand`
    - $I_i$: `Initial Inventory`
- No filtering or subsetting is applied; all records are included as per the query.

**Model Summary:**  
- Maximize total revenue from all products, choosing integer fulfillment quantities for each product, subject to each product’s demand and inventory limits.