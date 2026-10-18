##### Decision Variables

$x_i \geq 0$ (integer): Number of units of product $i$ to fulfill, for each product $i$.

##### Parameters

Let $P$ be the set of products:
\[
P = \{\text{sku\_I27},\ \text{sku\_I499},\ \text{sku\_I719},\ \text{sku\_T18},\ \text{sku\_T29},\ \text{sku\_T39},\ \text{sku\_T499},\ \text{sku\_T9},\ \text{sku\_3081},\ \text{sku\_339},\ \text{sku\_3799},\ \text{sku\_439},\ \text{sku\_539},\ \text{sku\_61399},\ \text{sku\_628},\ \text{sku\_708},\ \text{sku\_77},\ \text{sku\_79},\ \text{sku\_799},\ \text{sku\_8499},\ \text{sku\_89},\ \text{sku\_897},\ \text{sku\_9699},\ \text{sku\_bobo}\}
\]

For each product $i \in P$:

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

- Initial Inventory $I_i$:
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

##### Objective Function

\[
\max \sum_{i \in P} r_i x_i
\]

##### Constraints

1. Demand fulfillment: For each product $i \in P$,
   \[
   0 \leq x_i \leq \min\{d_i, I_i\}
   \]
   (You cannot fulfill more than the demand or available inventory for each product.)

2. Variable domain: $x_i$ integer, for all $i \in P$.

##### Full Model

\[
\begin{align*}
\max\ & \sum_{i \in P} r_i x_i \\
\text{s.t.}\quad
& 0 \leq x_i \leq \min\{d_i, I_i\},\quad \forall i \in P \\
& x_i \in \mathbb{Z}_+,\quad \forall i \in P
\end{align*}
\]

Where all parameters are as listed above.