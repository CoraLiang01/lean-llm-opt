##### Decision Variables

For each product $i \in P$:
- $x_i \geq 0$: Number of units of product $i$ to fulfill (integer or continuous, as appropriate).

##### Parameters

Let $P$ be the set of all products:

$P = \{$
sku_I27, sku_I499, sku_I719, sku_T18, sku_T29, sku_T39, sku_T499, sku_T9, sku_3081, sku_339, sku_3799, sku_439, sku_539, sku_61399, sku_628, sku_708, sku_77, sku_79, sku_799, sku_8499, sku_89, sku_897, sku_9699, sku_bobo
$\}$

For each $i \in P$:

- Revenue per unit $r_i$:
  - sku_I27: 238
  - sku_I499: 287
  - sku_I719: 268
  - sku_T18: 318
  - sku_T29: 207
  - sku_T39: 258
  - sku_T499: 249
  - sku_T9: 227
  - sku_3081: 198
  - sku_339: 254
  - sku_3799: 246
  - sku_439: 258
  - sku_539: 268
  - sku_61399: 278
  - sku_628: 268
  - sku_708: 298
  - sku_77: 258
  - sku_79: 315
  - sku_799: 264
  - sku_8499: 238
  - sku_89: 258
  - sku_897: 268
  - sku_9699: 288
  - sku_bobo: 228

- Initial inventory $I_i$:
  - sku_I27: 30
  - sku_I499: 20
  - sku_I719: 80
  - sku_T18: 70
  - sku_T29: 20
  - sku_T39: 160
  - sku_T499: 40
  - sku_T9: 10
  - sku_3081: 50
  - sku_339: 40
  - sku_3799: 90
  - sku_439: 10
  - sku_539: 20
  - sku_61399: 40
  - sku_628: 10
  - sku_708: 990
  - sku_77: 160
  - sku_79: 90
  - sku_799: 2870
  - sku_8499: 30
  - sku_89: 130
  - sku_897: 30
  - sku_9699: 170
  - sku_bobo: 170

- Demand $d_i$:
  - sku_I27: 6
  - sku_I499: 4
  - sku_I719: 16
  - sku_T18: 14
  - sku_T29: 4
  - sku_T39: 32
  - sku_T499: 8
  - sku_T9: 2
  - sku_3081: 10
  - sku_339: 8
  - sku_3799: 18
  - sku_439: 2
  - sku_539: 4
  - sku_61399: 8
  - sku_628: 2
  - sku_708: 198
  - sku_77: 32
  - sku_79: 18
  - sku_799: 570
  - sku_8499: 6
  - sku_89: 26
  - sku_897: 6
  - sku_9699: 33
  - sku_bobo: 33

##### Objective Function

\[
\max \sum_{i \in P} r_i x_i
\]

##### Constraints

1. Inventory and demand limits for each product:
   \[
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in P
   \]

##### Summary

- $x_i$: units of product $i$ fulfilled, $0 \leq x_i \leq \min\{I_i, d_i\}$
- Objective: maximize total revenue from fulfilled units
- All parameters (revenues, inventories, demands) are as listed above for each product.