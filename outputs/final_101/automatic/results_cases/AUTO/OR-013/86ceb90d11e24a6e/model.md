#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$ : set of all “4U” products (from column “Product Name” where prefix is “4U” in table_id file_0_view_0)

**Parameters:**
- $A_i$ : revenue per unit of product $i \in I$ (from column “Revenue” in table_id file_0_view_0)
- $d_i$ : total demand for product $i \in I$ (from column “Demand” in table_id file_0_view_0)
- $I_i$ : initial inventory for product $i \in I$ (from column “Initial Inventory” in table_id file_0_view_0)

**Decision Variables:**
- $x_i$ : number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in I
   \]
   (Equivalently, two constraints:)
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: file_0_view_0 (from OnlineSalesinUSA.csv)
    - Index set $I$: all rows where “Product Name” has prefix “4U”
    - Parameter $A_i$: column “Revenue”
    - Parameter $d_i$: column “Demand”
    - Parameter $I_i$: column “Initial Inventory”