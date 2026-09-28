##### Decision Variables

$x_i \geq 0$: number of units of product $i$ to order each day, for each $i \in P$ (continuous or integer, as appropriate).

##### Parameters

Let $P$ be the set of products:
- $P = \{$Spinach, Shiitake Mushrooms, Apples, Carrots, Basil, Potatoes, Green Beans, Blueberries, Oranges, Watermelons$\}$

For each product $i \in P$:
- $w_i$: weight per unit of product $i$
- $v_i$: value (profit) per unit of product $i$

Total stock capacity: $C = 875$

Product data (from 'products.csv'):
| ProductName           | $w_i$ | $v_i$ |
|-----------------------|-------|-------|
| Spinach               | 230   | 64    |
| Shiitake Mushrooms    | 637   | 75    |
| Apples                | 773   | 68    |
| Carrots               | 653   | 11    |
| Basil                 | 755   | 91    |
| Potatoes              | 670   | 31    |
| Green Beans           | 505   | 90    |
| Blueberries           | 821   | 56    |
| Oranges               | 83    | 10    |
| Watermelons           | 249   | 24    |

##### Objective Function

\[
\max \sum_{i \in P} v_i x_i
\]

##### Constraints

1. Stock capacity:
   \[
   \sum_{i \in P} w_i x_i \leq C
   \]
2. Nonnegativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]

##### Full Model (with parameters)

\[
\begin{align*}
\max\quad & 64x_{\text{Spinach}} + 75x_{\text{Shiitake Mushrooms}} + 68x_{\text{Apples}} + 11x_{\text{Carrots}} + 91x_{\text{Basil}} \\
& + 31x_{\text{Potatoes}} + 90x_{\text{Green Beans}} + 56x_{\text{Blueberries}} + 10x_{\text{Oranges}} + 24x_{\text{Watermelons}} \\
\text{s.t.}\quad & 230x_{\text{Spinach}} + 637x_{\text{Shiitake Mushrooms}} + 773x_{\text{Apples}} + 653x_{\text{Carrots}} + 755x_{\text{Basil}} \\
& + 670x_{\text{Potatoes}} + 505x_{\text{Green Beans}} + 821x_{\text{Blueberries}} + 83x_{\text{Oranges}} + 249x_{\text{Watermelons}} \leq 875 \\
& x_i \geq 0 \quad \forall i \in P
\end{align*}
\]

Where $x_i$ is the number of units of product $i$ to order each day.