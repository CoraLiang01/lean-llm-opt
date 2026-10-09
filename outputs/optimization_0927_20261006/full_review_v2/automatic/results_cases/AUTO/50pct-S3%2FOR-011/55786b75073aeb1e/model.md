Let $x_i$ be the number of units of product $i$ to order each day, where $i$ indexes the products in the order given in products.csv.

##### Sets and Parameters

- Let $I$ be the set of products, indexed in the order below:
    1. Spinach
    2. Shiitake Mushrooms
    3. Apples
    4. Carrots
    5. Basil
    6. Potatoes
    7. Green Beans
    8. Blueberries
    9. Oranges
    10. Watermelons

- For each product $i$:
    - $v_i$ = Value per unit (from "Value" column)
    - $w_i$ = Weight per unit (from "Weight" column)

- Let $C$ = 875 (overall stock capacity, from "Capacity" in capacity.csv)

##### Data

| $i$ | ProductName           | $v_i$ | $w_i$ |
|-----|-----------------------|-------|-------|
| 1   | Spinach               | 64    | 230   |
| 2   | Shiitake Mushrooms    | 75    | 637   |
| 3   | Apples                | 68    | 773   |
| 4   | Carrots               | 11    | 653   |
| 5   | Basil                 | 91    | 755   |
| 6   | Potatoes              | 31    | 670   |
| 7   | Green Beans           | 90    | 505   |
| 8   | Blueberries           | 56    | 821   |
| 9   | Oranges               | 10    | 83    |
| 10  | Watermelons           | 24    | 249   |

##### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

##### Objective

$\max \sum_{i=1}^{10} v_i x_i$

That is,

$\max \left(64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10}\right)$

##### Constraints

1. **Overall Stock Capacity:**

$\sum_{i=1}^{10} w_i x_i \leq 875$

That is,

$230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875$

2. **Non-negativity and Integrality:**

$x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10$

##### Complete Model

\[
\begin{align*}
\max \quad & 64x_1 + 75x_2 + 68x_3 + 11x_4 + 91x_5 + 31x_6 + 90x_7 + 56x_8 + 10x_9 + 24x_{10} \\
\text{s.t.} \quad & 230x_1 + 637x_2 + 773x_3 + 653x_4 + 755x_5 + 670x_6 + 505x_7 + 821x_8 + 83x_9 + 249x_{10} \leq 875 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad i = 1, \ldots, 10
\end{align*}
\]

Where the mapping of $i$ to ProductName is as listed above.