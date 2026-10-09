Let $x_i$ be the number of units of product $i$ to fulfill, for each product $i$ listed below.

**Objective:**
\[
\max \sum_{i} \text{Revenue}_i \cdot x_i
\]

**Subject to:**

For each product $i$ (using the "Product Name" as $i$):

1. **Demand fulfillment cannot exceed demand or available inventory:**
   \[
   0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}
   \]
   (Since both are upper bounds, but demand is always less than or equal to inventory in this data, you may use $x_i \leq \text{Demand}_i$ and $x_i \leq \text{Initial Inventory}_i$ separately.)

2. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}
   \]

**Parameters (from data, in source order):**

| Product Name    | Revenue | Demand | Initial Inventory |
|-----------------|---------|--------|------------------|
| sku_I27         | 238     | 6      | 30               |
| sku_I499        | 287     | 4      | 20               |
| sku_I719        | 268     | 16     | 80               |
| sku_T18         | 318     | 14     | 70               |
| sku_T29         | 207     | 4      | 20               |
| sku_T39         | 258     | 32     | 160              |
| sku_T499        | 249     | 8      | 40               |
| sku_T9          | 227     | 2      | 10               |
| sku_3081        | 198     | 10     | 50               |
| sku_339         | 254     | 8      | 40               |
| sku_3799        | 246     | 18     | 90               |
| sku_439         | 258     | 2      | 10               |
| sku_539         | 268     | 4      | 20               |
| sku_61399       | 278     | 8      | 40               |
| sku_628         | 268     | 2      | 10               |
| sku_708         | 298     | 198    | 990              |
| sku_77          | 258     | 32     | 160              |
| sku_79          | 315     | 18     | 90               |
| sku_799         | 264     | 570    | 2870             |
| sku_8499        | 238     | 6      | 30               |
| sku_89          | 258     | 26     | 130              |
| sku_897         | 268     | 6      | 30               |
| sku_9699        | 288     | 33     | 170              |
| sku_bobo        | 228     | 33     | 170              |

**Full Model:**

\[
\max \left(
238x_{\text{sku\_I27}} +
287x_{\text{sku\_I499}} +
268x_{\text{sku\_I719}} +
318x_{\text{sku\_T18}} +
207x_{\text{sku\_T29}} +
258x_{\text{sku\_T39}} +
249x_{\text{sku\_T499}} +
227x_{\text{sku\_T9}} +
198x_{\text{sku\_3081}} +
254x_{\text{sku\_339}} +
246x_{\text{sku\_3799}} +
258x_{\text{sku\_439}} +
268x_{\text{sku\_539}} +
278x_{\text{sku\_61399}} +
268x_{\text{sku\_628}} +
298x_{\text{sku\_708}} +
258x_{\text{sku\_77}} +
315x_{\text{sku\_79}} +
264x_{\text{sku\_799}} +
238x_{\text{sku\_8499}} +
258x_{\text{sku\_89}} +
268x_{\text{sku\_897}} +
288x_{\text{sku\_9699}} +
228x_{\text{sku\_bobo}}
\right)
\]

Subject to, for each product (row):

\[
\begin{align*}
0 \leq x_{\text{sku\_I27}} &\leq 6 \\
0 \leq x_{\text{sku\_I499}} &\leq 4 \\
0 \leq x_{\text{sku\_I719}} &\leq 16 \\
0 \leq x_{\text{sku\_T18}} &\leq 14 \\
0 \leq x_{\text{sku\_T29}} &\leq 4 \\
0 \leq x_{\text{sku\_T39}} &\leq 32 \\
0 \leq x_{\text{sku\_T499}} &\leq 8 \\
0 \leq x_{\text{sku\_T9}} &\leq 2 \\
0 \leq x_{\text{sku\_3081}} &\leq 10 \\
0 \leq x_{\text{sku\_339}} &\leq 8 \\
0 \leq x_{\text{sku\_3799}} &\leq 18 \\
0 \leq x_{\text{sku\_439}} &\leq 2 \\
0 \leq x_{\text{sku\_539}} &\leq 4 \\
0 \leq x_{\text{sku\_61399}} &\leq 8 \\
0 \leq x_{\text{sku\_628}} &\leq 2 \\
0 \leq x_{\text{sku\_708}} &\leq 198 \\
0 \leq x_{\text{sku\_77}} &\leq 32 \\
0 \leq x_{\text{sku\_79}} &\leq 18 \\
0 \leq x_{\text{sku\_799}} &\leq 570 \\
0 \leq x_{\text{sku\_8499}} &\leq 6 \\
0 \leq x_{\text{sku\_89}} &\leq 26 \\
0 \leq x_{\text{sku\_897}} &\leq 6 \\
0 \leq x_{\text{sku\_9699}} &\leq 33 \\
0 \leq x_{\text{sku\_bobo}} &\leq 33 \\
x_i \in \mathbb{Z}_{\geq 0} &\quad \forall i
\end{align*}
\]

Where $x_i$ is the number of units of product $i$ to fulfill, for each product $i$ as listed above.