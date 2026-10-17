##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order daily, for each $i \in P$ (where $P$ is the set of all produce types).

##### Parameters

- $P = \{$Spinach, Shiitake Mushrooms, Apples, Carrots, Basil, Potatoes, Green Beans, Blueberries, Oranges, Watermelons$\}$
- Per-unit weights $w_i$ and per-unit values $v_i$ for each $i \in P$:

| Product Name         | $w_i$ (Weight) | $v_i$ (Value) |
|---------------------|:--------------:|:-------------:|
| Spinach             | 282            | 49            |
| Shiitake Mushrooms  | 83             | 30            |
| Apples              | 251            | 30            |
| Carrots             | 257            | 18            |
| Basil               | 88             | 54            |
| Potatoes            | 52             | 27            |
| Green Beans         | 198            | 91            |
| Blueberries         | 203            | 88            |
| Oranges             | 87             | 78            |
| Watermelons         | 265            | 22            |

- Total inventory capacity: $C = 1035$

##### Objective Function

\[
\max \sum_{i \in P} v_i x_i
\]

##### Constraints

1. Inventory capacity constraint:
   \[
   \sum_{i \in P} w_i x_i \leq 1035
   \]
2. Integer and nonnegativity constraints:
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in P
   \]

##### Full Model

\[
\begin{align*}
\max\quad & 49x_{\text{Spinach}} + 30x_{\text{Shiitake Mushrooms}} + 30x_{\text{Apples}} + 18x_{\text{Carrots}} + 54x_{\text{Basil}} \\
& + 27x_{\text{Potatoes}} + 91x_{\text{Green Beans}} + 88x_{\text{Blueberries}} + 78x_{\text{Oranges}} + 22x_{\text{Watermelons}} \\
\text{s.t.}\quad & 282x_{\text{Spinach}} + 83x_{\text{Shiitake Mushrooms}} + 251x_{\text{Apples}} + 257x_{\text{Carrots}} + 88x_{\text{Basil}} \\
& + 52x_{\text{Potatoes}} + 198x_{\text{Green Beans}} + 203x_{\text{Blueberries}} + 87x_{\text{Oranges}} + 265x_{\text{Watermelons}} \leq 1035 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in P
\end{align*}
\]

##### Retrieved Information

- Capacity: $C = 1035$
- Products, weights, and values:

  - Spinach: weight 282, value 49
  - Shiitake Mushrooms: weight 83, value 30
  - Apples: weight 251, value 30
  - Carrots: weight 257, value 18
  - Basil: weight 88, value 54
  - Potatoes: weight 52, value 27
  - Green Beans: weight 198, value 91
  - Blueberries: weight 203, value 88
  - Oranges: weight 87, value 78
  - Watermelons: weight 265, value 22