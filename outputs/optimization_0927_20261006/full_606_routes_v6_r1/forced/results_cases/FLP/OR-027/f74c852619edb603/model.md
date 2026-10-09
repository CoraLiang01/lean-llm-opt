##### Parameters

Let $I$ be the set of products classified as ‘Organ’:
$$
I = \{\text{Organic Fruits},\ \text{Organic Staples},\ \text{Organic Vegetables}\}
$$

For each $i \in I$:

- Revenue per unit: 
  - $\text{Organic Fruits}: r_{\text{Organic Fruits}} = 60.8$
  - $\text{Organic Staples}: r_{\text{Organic Staples}} = 918.45$
  - $\text{Organic Vegetables}: r_{\text{Organic Vegetables}} = 77.52$

- Initial Inventory:
  - $\text{Organic Fruits}: s_{\text{Organic Fruits}} = 5,\!034,\!020.0$
  - $\text{Organic Staples}: s_{\text{Organic Staples}} = 5,\!589,\!290.0$
  - $\text{Organic Vegetables}: s_{\text{Organic Vegetables}} = 5,\!202,\!710.0$

- Demand:
  - $\text{Organic Fruits}: d_{\text{Organic Fruits}} = 678,\!906$
  - $\text{Organic Staples}: d_{\text{Organic Staples}} = 749,\!927$
  - $\text{Organic Vegetables}: d_{\text{Organic Vegetables}} = 699,\!808$

##### Decision Variables

For each $i \in I$:

- $x_i \geq 0$: number of units of product $i$ to fulfill.

##### Objective Function

$$
\max \sum_{i \in I} r_i x_i
$$

##### Constraints

For each $i \in I$:

1. Inventory constraint: $x_i \leq s_i$
2. Demand constraint: $x_i \leq d_i$
3. Nonnegativity: $x_i \geq 0$

##### Complete Model

$$
\begin{align*}
\max\quad & 60.8\, x_{\text{Organic Fruits}} + 918.45\, x_{\text{Organic Staples}} + 77.52\, x_{\text{Organic Vegetables}} \\
\text{s.t.}\quad 
& x_{\text{Organic Fruits}} \leq 5,\!034,\!020.0 \\
& x_{\text{Organic Fruits}} \leq 678,\!906 \\
& x_{\text{Organic Staples}} \leq 5,\!589,\!290.0 \\
& x_{\text{Organic Staples}} \leq 749,\!927 \\
& x_{\text{Organic Vegetables}} \leq 5,\!202,\!710.0 \\
& x_{\text{Organic Vegetables}} \leq 699,\!808 \\
& x_{\text{Organic Fruits}} \geq 0,\quad x_{\text{Organic Staples}} \geq 0,\quad x_{\text{Organic Vegetables}} \geq 0
\end{align*}
$$

##### Retrieved Information

- Products: Organic Fruits, Organic Staples, Organic Vegetables
- Revenue: 60.8, 918.45, 77.52
- Initial Inventory: 5,034,020.0; 5,589,290.0; 5,202,710.0
- Demand: 678,906; 749,927; 699,808