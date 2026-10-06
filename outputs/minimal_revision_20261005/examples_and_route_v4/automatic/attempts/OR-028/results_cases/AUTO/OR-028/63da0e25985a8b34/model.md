**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of products (Product Name from WomenClothingEcommerceSalesData.csv)

**Parameters**
- $r_i$: Revenue per unit of product $i$ (Revenue, table_id: file_0_view_0, column: Revenue)
- $d_i$: Demand for product $i$ (Demand, table_id: file_0_view_0, column: Demand)
- $s_i$: Initial inventory for product $i$ (Initial Inventory, table_id: file_0_view_0, column: Initial Inventory)

**Decision Variables**
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

**Objective**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints**
1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Cannot fulfill more than available inventory:**
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
3. **Nonnegativity and integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All records in WomenClothingEcommerceSalesData.csv, column Product Name, table_id: file_0_view_0
- $r_i$: WomenClothingEcommerceSalesData.csv, column Revenue, table_id: file_0_view_0
- $d_i$: WomenClothingEcommerceSalesData.csv, column Demand, table_id: file_0_view_0
- $s_i$: WomenClothingEcommerceSalesData.csv, column Initial Inventory, table_id: file_0_view_0