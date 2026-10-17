#### Abstract Optimization Model

**Index Sets:**
- $I$ : Set of car models classified under ‘FDK57’.  
  $ I = \{ i \mid \text{‘Product Name’ contains ‘FDK57’ in BigMartSales.csv} \} $

**Parameters:**
- $a_i$ : Revenue per unit for car model $i \in I$ (from column ‘Revenue’ in BigMartSales.csv)
- $d_i$ : Deterministic demand for car model $i \in I$ (from column ‘Demand’ in BigMartSales.csv)
- $s_i$ : Initial inventory for car model $i \in I$ (from column ‘Initial Inventory’ in BigMartSales.csv)

**Decision Variables:**
- $x_i$ : Number of units of car model $i \in I$ to fulfill (integer, $x_i \geq 0$)

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. **Inventory Bound:**  
  $x_i \leq s_i \quad \forall i \in I$

2. **Demand Bound:**  
  $x_i \leq d_i \quad \forall i \in I$

3. **Nonnegativity and Integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

**Data Mapping:**

- **Source Table:** `BigMartSales.csv` (table_id: `file_0_view_0`)
- **Index Set $I$:** All rows where `Product Name` contains the substring `FDK57`
- **Parameter $a_i$:** Column `Revenue` for each $i \in I$
- **Parameter $d_i$:** Column `Demand` for each $i \in I$
- **Parameter $s_i$:** Column `Initial Inventory` for each $i \in I`
- **Decision Variable $x_i$:** Quantity fulfilled for each $i \in I$

---

**Summary:**  
Maximize total revenue from fulfilling sales of ‘FDK57’ car models, subject to initial inventory and deterministic demand limits, with integer, nonnegative fulfillment decisions for each eligible model. All data is mapped directly from the specified columns and subset in `BigMartSales.csv`.