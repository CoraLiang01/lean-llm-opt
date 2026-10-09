Abstract Mathematical Optimization Model

Index Sets:
- \( I \): Set of all “4U” products, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (Source: OnlineSalesinUSA.csv, column "Revenue")
- \( d_i \): Demand for product \( i \) during the sales horizon. (Source: OnlineSalesinUSA.csv, column "Demand")
- \( s_i \): Initial inventory of product \( i \). (Source: OnlineSalesinUSA.csv, column "Initial Inventory")

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers), for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory and demand fulfillment limits:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \in I
   \]
   (Equivalently, two constraints per product:)
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
   \[
   x_i \leq d_i \quad \forall i \in I
   \]

2. Integer and non-negativity:
   \[
   x_i \in \mathbb{Z}_+ \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All rows in OnlineSalesinUSA.csv where "Product Name" starts with "4U" (table_id: file_0_view_0, column: "Product Name").
- Parameter \( r_i \): OnlineSalesinUSA.csv, column "Revenue", table_id: file_0_view_0.
- Parameter \( d_i \): OnlineSalesinUSA.csv, column "Demand", table_id: file_0_view_0.
- Parameter \( s_i \): OnlineSalesinUSA.csv, column "Initial Inventory", table_id: file_0_view_0.