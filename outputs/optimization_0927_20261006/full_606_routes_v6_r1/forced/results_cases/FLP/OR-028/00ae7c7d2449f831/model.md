##### Sets and Indices

Let $I$ be the set of products:
$$
I = \{\text{sku\_I27},\ \text{sku\_I499},\ \text{sku\_I719},\ \text{sku\_T18},\ \text{sku\_T29},\ \text{sku\_T39},\ \text{sku\_T499},\ \text{sku\_T9},\ \text{sku\_3081},\ \text{sku\_339},\ \text{sku\_3799},\ \text{sku\_439},\ \text{sku\_539},\ \text{sku\_61399},\ \text{sku\_628},\ \text{sku\_708},\ \text{sku\_77},\ \text{sku\_79},\ \text{sku\_799},\ \text{sku\_8499},\ \text{sku\_89},\ \text{sku\_897},\ \text{sku\_9699},\ \text{sku\_bobo}\}
$$

##### Parameters

For each product $i \in I$:

- Revenue per unit $r_i$:
  - sku_I27: $r_{\text{sku\_I27}} = 238$
  - sku_I499: $r_{\text{sku\_I499}} = 287$
  - sku_I719: $r_{\text{sku\_I719}} = 268$
  - sku_T18: $r_{\text{sku\_T18}} = 318$
  - sku_T29: $r_{\text{sku\_T29}} = 207$
  - sku_T39: $r_{\text{sku\_T39}} = 258$
  - sku_T499: $r_{\text{sku\_T499}} = 249$
  - sku_T9: $r_{\text{sku\_T9}} = 227$
  - sku_3081: $r_{\text{sku\_3081}} = 198$
  - sku_339: $r_{\text{sku\_339}} = 254$
  - sku_3799: $r_{\text{sku\_3799}} = 246$
  - sku_439: $r_{\text{sku\_439}} = 258$
  - sku_539: $r_{\text{sku\_539}} = 268$
  - sku_61399: $r_{\text{sku\_61399}} = 278$
  - sku_628: $r_{\text{sku\_628}} = 268$
  - sku_708: $r_{\text{sku\_708}} = 298$
  - sku_77: $r_{\text{sku\_77}} = 258$
  - sku_79: $r_{\text{sku\_79}} = 315$
  - sku_799: $r_{\text{sku\_799}} = 264$
  - sku_8499: $r_{\text{sku\_8499}} = 238$
  - sku_89: $r_{\text{sku\_89}} = 258$
  - sku_897: $r_{\text{sku\_897}} = 268$
  - sku_9699: $r_{\text{sku\_9699}} = 288$
  - sku_bobo: $r_{\text{sku\_bobo}} = 228$

- Demand $d_i$:
  - sku_I27: $d_{\text{sku\_I27}} = 6$
  - sku_I499: $d_{\text{sku\_I499}} = 4$
  - sku_I719: $d_{\text{sku\_I719}} = 16$
  - sku_T18: $d_{\text{sku\_T18}} = 14$
  - sku_T29: $d_{\text{sku\_T29}} = 4$
  - sku_T39: $d_{\text{sku\_T39}} = 32$
  - sku_T499: $d_{\text{sku\_T499}} = 8$
  - sku_T9: $d_{\text{sku\_T9}} = 2$
  - sku_3081: $d_{\text{sku\_3081}} = 10$
  - sku_339: $d_{\text{sku\_339}} = 8$
  - sku_3799: $d_{\text{sku\_3799}} = 18$
  - sku_439: $d_{\text{sku\_439}} = 2$
  - sku_539: $d_{\text{sku\_539}} = 4$
  - sku_61399: $d_{\text{sku\_61399}} = 8$
  - sku_628: $d_{\text{sku\_628}} = 2$
  - sku_708: $d_{\text{sku\_708}} = 198$
  - sku_77: $d_{\text{sku\_77}} = 32$
  - sku_79: $d_{\text{sku\_79}} = 18$
  - sku_799: $d_{\text{sku\_799}} = 570$
  - sku_8499: $d_{\text{sku\_8499}} = 6$
  - sku_89: $d_{\text{sku\_89}} = 26$
  - sku_897: $d_{\text{sku\_897}} = 6$
  - sku_9699: $d_{\text{sku\_9699}} = 33$
  - sku_bobo: $d_{\text{sku\_bobo}} = 33$

- Initial Inventory $s_i$:
  - sku_I27: $s_{\text{sku\_I27}} = 30$
  - sku_I499: $s_{\text{sku\_I499}} = 20$
  - sku_I719: $s_{\text{sku\_I719}} = 80$
  - sku_T18: $s_{\text{sku\_T18}} = 70$
  - sku_T29: $s_{\text{sku\_T29}} = 20$
  - sku_T39: $s_{\text{sku\_T39}} = 160$
  - sku_T499: $s_{\text{sku\_T499}} = 40$
  - sku_T9: $s_{\text{sku\_T9}} = 10$
  - sku_3081: $s_{\text{sku\_3081}} = 50$
  - sku_339: $s_{\text{sku\_339}} = 40$
  - sku_3799: $s_{\text{sku\_3799}} = 90$
  - sku_439: $s_{\text{sku\_439}} = 10$
  - sku_539: $s_{\text{sku\_539}} = 20$
  - sku_61399: $s_{\text{sku\_61399}} = 40$
  - sku_628: $s_{\text{sku\_628}} = 10$
  - sku_708: $s_{\text{sku\_708}} = 990$
  - sku_77: $s_{\text{sku\_77}} = 160$
  - sku_79: $s_{\text{sku\_79}} = 90$
  - sku_799: $s_{\text{sku\_799}} = 2870$
  - sku_8499: $s_{\text{sku\_8499}} = 30$
  - sku_89: $s_{\text{sku\_89}} = 130$
  - sku_897: $s_{\text{sku\_897}} = 30$
  - sku_9699: $s_{\text{sku\_9699}} = 170$
  - sku_bobo: $s_{\text{sku\_bobo}} = 170$

##### Decision Variables

For each $i \in I$:

- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$

##### Objective Function

Maximize total revenue:
$$
\max \sum_{i \in I} r_i x_i
$$

##### Constraints

For each $i \in I$:

1. Cannot fulfill more than demand:
   $$
   x_i \leq d_i
   $$
2. Cannot fulfill more than available inventory:
   $$
   x_i \leq s_i
   $$
3. Nonnegativity and integrality:
   $$
   x_i \geq 0,\quad x_i \in \mathbb{Z}
   $$

##### Summary Table

| Product Name   | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|----------------|-----------------|---------------|---------------------------|
| sku_I27        | 238             | 6             | 30                        |
| sku_I499       | 287             | 4             | 20                        |
| sku_I719       | 268             | 16            | 80                        |
| sku_T18        | 318             | 14            | 70                        |
| sku_T29        | 207             | 4             | 20                        |
| sku_T39        | 258             | 32            | 160                       |
| sku_T499       | 249             | 8             | 40                        |
| sku_T9         | 227             | 2             | 10                        |
| sku_3081       | 198             | 10            | 50                        |
| sku_339        | 254             | 8             | 40                        |
| sku_3799       | 246             | 18            | 90                        |
| sku_439        | 258             | 2             | 10                        |
| sku_539        | 268             | 4             | 20                        |
| sku_61399      | 278             | 8             | 40                        |
| sku_628        | 268             | 2             | 10                        |
| sku_708        | 298             | 198           | 990                       |
| sku_77         | 258             | 32            | 160                       |
| sku_79         | 315             | 18            | 90                        |
| sku_799        | 264             | 570           | 2870                      |
| sku_8499       | 238             | 6             | 30                        |
| sku_89         | 258             | 26            | 130                       |
| sku_897        | 268             | 6             | 30                        |
| sku_9699       | 288             | 33            | 170                       |
| sku_bobo       | 228             | 33            | 170                       |

##### Complete Model

$$
\begin{align*}
\max\ & \sum_{i \in I} r_i x_i \\
\text{s.t.}\quad
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \geq 0,\ x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
$$

Where all parameters are as listed above.