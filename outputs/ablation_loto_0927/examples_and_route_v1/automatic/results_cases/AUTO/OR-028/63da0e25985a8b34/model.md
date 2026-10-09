Let $x_i$ be the number of units of product $i$ to fulfill, for each product in the table below.

**Objective:**
\[
\max \sum_{i} \text{Revenue}_i \cdot x_i
\]

**Subject to:**

- Demand fulfillment cannot exceed demand or available inventory:
  \[
  0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\} \qquad \forall i
  \]
  (Since both demand and inventory are upper bounds, $x_i$ cannot exceed either.)

- $x_i$ are nonnegative integers:
  \[
  x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i
  \]

**Parameters (from data):**

| Product Name | Revenue | Demand | Initial Inventory |
|--------------|---------|--------|------------------|
| sku_I27      | 238     | 6      | 30               |
| sku_I499     | 287     | 4      | 20               |
| sku_I719     | 268     | 16     | 80               |
| sku_T18      | 318     | 14     | 70               |
| sku_T29      | 207     | 4      | 20               |
| sku_T39      | 258     | 32     | 160              |
| sku_T499     | 249     | 8      | 40               |
| sku_T9       | 227     | 2      | 10               |
| sku_3081     | 198     | 10     | 50               |
| sku_339      | 254     | 8      | 40               |
| sku_3799     | 246     | 18     | 90               |
| sku_439      | 258     | 2      | 10               |
| sku_539      | 268     | 4      | 20               |
| sku_61399    | 278     | 8      | 40               |
| sku_628      | 268     | 2      | 10               |
| sku_708      | 298     | 198    | 990              |
| sku_77       | 258     | 32     | 160              |
| sku_79       | 315     | 18     | 90               |
| sku_799      | 264     | 570    | 2870             |
| sku_8499     | 238     | 6      | 30               |
| sku_89       | 258     | 26     | 130              |
| sku_897      | 268     | 6      | 30               |
| sku_9699     | 288     | 33     | 170              |
| sku_bobo     | 228     | 33     | 170              |

**Explicitly, for each product $i$ (row above):**
\[
0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}, \quad x_i \in \mathbb{Z}_{\geq 0}
\]

**Maximize:**
\[
238x_{\text{sku\_I27}} + 287x_{\text{sku\_I499}} + 268x_{\text{sku\_I719}} + 318x_{\text{sku\_T18}} + 207x_{\text{sku\_T29}} + 258x_{\text{sku\_T39}} + 249x_{\text{sku\_T499}} + 227x_{\text{sku\_T9}} + 198x_{\text{sku\_3081}} + 254x_{\text{sku\_339}} + 246x_{\text{sku\_3799}} + 258x_{\text{sku\_439}} + 268x_{\text{sku\_539}} + 278x_{\text{sku\_61399}} + 268x_{\text{sku\_628}} + 298x_{\text{sku\_708}} + 258x_{\text{sku\_77}} + 315x_{\text{sku\_79}} + 264x_{\text{sku\_799}} + 238x_{\text{sku\_8499}} + 258x_{\text{sku\_89}} + 268x_{\text{sku\_897}} + 288x_{\text{sku\_9699}} + 228x_{\text{sku\_bobo}}
\]

**Subject to, for each $i$:**
\[
0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}, \quad x_i \in \mathbb{Z}_{\geq 0}
\]