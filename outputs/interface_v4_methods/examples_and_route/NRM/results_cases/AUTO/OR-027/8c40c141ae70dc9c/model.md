#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of products classified under ‘Organ’ (indexed by $i$; see Data Mapping).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from Revenue column).
- $s_i$: Initial inventory of product $i$ (from Initial Inventory column).
- $d_i$: Demand for product $i$ (from Demand column).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill; $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory constraint:** 
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. **Demand constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Nonnegativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$: All records in SupermartGrocerySales-RetailAnalyticsDataset.csv where Sub Category starts with or contains "Organ".
- $r_i$: SupermartGrocerySales-RetailAnalyticsDataset.csv, column Revenue, for each $i$.
- $s_i$: SupermartGrocerySales-RetailAnalyticsDataset.csv, column Initial Inventory, for each $i$.
- $d_i$: SupermartGrocerySales-RetailAnalyticsDataset.csv, column Demand, for each $i$.
- $x_i$: Decision variable for each $i$ as above.

**Table Reference:**  
SupermartGrocerySales-RetailAnalyticsDataset.csv, columns: Sub Category, Revenue, Initial Inventory, Demand, rows:  
- source_row 17: Organic Fruits  
- source_row 18: Organic Staples  
- source_row 19: Organic Vegetables

**Note:** Each $i$ is identified by the Sub Category value in the table.