##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: number of units of produce type $i$ to order daily, for each $i \in P$.

##### Parameters

Let $P$ be the set of produce types:
$$
P = \{\text{Spinach},\ \text{Shiitake Mushrooms},\ \text{Apples},\ \text{Carrots},\ \text{Basil},\ \text{Potatoes},\ \text{Green Beans},\ \text{Blueberries},\ \text{Oranges},\ \text{Watermelons}\}
$$

Per-unit weights $w_i$ and values $v_i$ for each $i \in P$:

| Produce               | $w_i$ (Weight) | $v_i$ (Value) |
|-----------------------|:--------------:|:-------------:|
| Spinach               | 282            | 49            |
| Shiitake Mushrooms    | 83             | 30            |
| Apples                | 251            | 30            |
| Carrots               | 257            | 18            |
| Basil                 | 88             | 54            |
| Potatoes              | 52             | 27            |
| Green Beans           | 198            | 91            |
| Blueberries           | 203            | 88            |
| Oranges               | 87             | 78            |
| Watermelons           | 265            | 22            |

Total capacity: $C = 1035$

##### Objective Function

$$
\max \sum_{i \in P} v_i x_i
$$

##### Constraints

1. Capacity constraint:
   $$
   \sum_{i \in P} w_i x_i \leq 1035
   $$
2. Integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in P
   $$

##### Full Model

$$
\begin{align*}
\max\quad & 49x_{\text{Spinach}} + 30x_{\text{Shiitake Mushrooms}} + 30x_{\text{Apples}} + 18x_{\text{Carrots}} + 54x_{\text{Basil}} \\
& + 27x_{\text{Potatoes}} + 91x_{\text{Green Beans}} + 88x_{\text{Blueberries}} + 78x_{\text{Oranges}} + 22x_{\text{Watermelons}} \\
\text{s.t.}\quad & 282x_{\text{Spinach}} + 83x_{\text{Shiitake Mushrooms}} + 251x_{\text{Apples}} + 257x_{\text{Carrots}} + 88x_{\text{Basil}} \\
& + 52x_{\text{Potatoes}} + 198x_{\text{Green Beans}} + 203x_{\text{Blueberries}} + 87x_{\text{Oranges}} + 265x_{\text{Watermelons}} \leq 1035 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in P
\end{align*}
$$

All parameters and vectors are as retrieved from the CSV files.