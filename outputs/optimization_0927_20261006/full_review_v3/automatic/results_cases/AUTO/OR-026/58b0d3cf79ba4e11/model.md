#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all products classified as ‘Fashion’ (from SupermarketSales.csv, see Data Mapping).

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$ (SupermarketSales.csv, column "Revenue").
- $d_i$: Demand for product $i \in I$ (SupermarketSales.csv, column "Demand").
- $I_i$: Initial inventory for product $i \in I$ (SupermarketSales.csv, column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{I_i, d_i\} \quad \forall i \in I
   \]

**Data Mapping:**
- Table: SupermarketSales.csv
- Table ID: file_0_view_0
- Filter: All rows where "Product Name" has prefix "Fashion" (i.e., products classified as ‘Fashion’)
- Columns used: "Product Name", "Revenue", "Initial Inventory", "Demand"