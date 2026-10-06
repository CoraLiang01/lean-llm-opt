**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of all products classified under ‘Fashion’, indexed by $i$. (From SupermarketSales.csv, all rows with "Product Name" starting with "Fashion")
  
**Parameters:**
- $r_i$: Revenue per unit of product $i$. (SupermarketSales.csv, column "Revenue")
- $d_i$: Demand for product $i$. (SupermarketSales.csv, column "Demand")
- $s_i$: Initial inventory of product $i$. (SupermarketSales.csv, column "Initial Inventory")

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill. ($x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$)

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
2. **Inventory availability:**  
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]
3. **Nonnegativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All records in SupermarketSales.csv where "Product Name" starts with "Fashion" (see table_id: file_0_view_0, column "Product Name")
- $r_i$: SupermarketSales.csv, table_id: file_0_view_0, column "Revenue", for each $i$
- $d_i$: SupermarketSales.csv, table_id: file_0_view_0, column "Demand", for each $i$
- $s_i$: SupermarketSales.csv, table_id: file_0_view_0, column "Initial Inventory", for each $i$