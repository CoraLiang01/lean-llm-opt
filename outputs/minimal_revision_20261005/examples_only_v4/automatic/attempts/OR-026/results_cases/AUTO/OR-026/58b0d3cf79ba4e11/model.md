**Mathematical Optimization Model (Abstract Formulation)**

**Index Sets:**
- \( I \): Set of all products \( i \) classified as 'Fashion' (i.e., all products in "SupermarketSales.csv" where "Product Name" contains 'Fashion accessories_').

**Parameters:**
- \( r_i \): Revenue per unit of product \( i \).  
  Source: "SupermarketSales.csv", column "Revenue", for \( i \in I \).
- \( d_i \): Deterministic demand for product \( i \).  
  Source: "SupermarketSales.csv", column "Demand", for \( i \in I \).
- \( s_i \): Initial inventory available for product \( i \).  
  Source: "SupermarketSales.csv", column "Initial Inventory", for \( i \in I \).

**Decision Variables:**
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), for all \( i \in I \).

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled 'Fashion' products.)

**Constraints:**
1. **Demand and Inventory Fulfillment:**
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]
   (Cannot fulfill more than available inventory or demand for each product.)

**Data Mapping:**
- Table: "SupermarketSales.csv", table_id: file_0_view_0
  - Index set \( I \): All rows where "Product Name" contains 'Fashion accessories_'
  - \( r_i \): "Revenue"
  - \( d_i \): "Demand"
  - \( s_i \): "Initial Inventory"

**Notes:**
- All parameters are directly mapped from the specified columns for the filtered set of 'Fashion' products.
- The model is a deterministic, single-period fulfillment problem for the 'Fashion' category only.