Let $x_i$ be the number of units of product $i$ to fulfill demand for product $i$.

Objective:
$$
\max \sum_{i} \text{Revenue}_i \cdot x_i
$$

Subject to, for each product $i$ (using the Product Name as $i$):

1. Demand fulfillment cannot exceed demand or available inventory:
$$
0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}
$$

2. $x_i$ are nonnegative integers:
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

Where the data is:

| Product Name | Revenue | Demand | Initial Inventory |
|-------------|---------|--------|------------------|
| sku_I27     | 238     | 6      | 30               |
| sku_I499    | 287     | 4      | 20               |
| sku_I719    | 268     | 16     | 80               |
| sku_T18     | 318     | 14     | 70               |
| sku_T29     | 207     | 4      | 20               |
| sku_T39     | 258     | 32     | 160              |
| sku_T499    | 249     | 8      | 40               |
| sku_T9      | 227     | 2      | 10               |
| sku_3081    | 198     | 10     | 50               |
| sku_339     | 254     | 8      | 40               |
| sku_3799    | 246     | 18     | 90               |
| sku_439     | 258     | 2      | 10               |
| sku_539     | 268     | 4      | 20               |
| sku_61399   | 278     | 8      | 40               |
| sku_628     | 268     | 2      | 10               |
| sku_708     | 298     | 198    | 990              |
| sku_77      | 258     | 32     | 160              |
| sku_79      | 315     | 18     | 90               |
| sku_799     | 264     | 570    | 2870             |
| sku_8499    | 238     | 6      | 30               |
| sku_89      | 258     | 26     | 130              |
| sku_897     | 268     | 6      | 30               |
| sku_9699    | 288     | 33     | 170              |
| sku_bobo    | 228     | 33     | 170              |

Explicitly, for each $i$ (Product Name):

$$
\max \left[
238 x_{\text{sku\_I27}} + 287 x_{\text{sku\_I499}} + 268 x_{\text{sku\_I719}} + 318 x_{\text{sku\_T18}} + 207 x_{\text{sku\_T29}} + 258 x_{\text{sku\_T39}} + 249 x_{\text{sku\_T499}} + 227 x_{\text{sku\_T9}} + 198 x_{\text{sku\_3081}} + 254 x_{\text{sku\_339}} + 246 x_{\text{sku\_3799}} + 258 x_{\text{sku\_439}} + 268 x_{\text{sku\_539}} + 278 x_{\text{sku\_61399}} + 268 x_{\text{sku\_628}} + 298 x_{\text{sku\_708}} + 258 x_{\text{sku\_77}} + 315 x_{\text{sku\_79}} + 264 x_{\text{sku\_799}} + 238 x_{\text{sku\_8499}} + 258 x_{\text{sku\_89}} + 268 x_{\text{sku\_897}} + 288 x_{\text{sku\_9699}} + 228 x_{\text{sku\_bobo}}
\right]
$$

Subject to, for each product $i$:

$$
0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}
$$

and

$$
x_i \in \mathbb{Z}_{\geq 0}
$$

for all $i$ in the table above.