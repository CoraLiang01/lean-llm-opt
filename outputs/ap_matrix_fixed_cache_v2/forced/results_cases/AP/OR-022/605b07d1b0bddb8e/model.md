##### Decision Variables

Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘27in’ products) to be fulfilled.

##### Objective Function

$\max \left( 261.2933\, x_1 + 52.4965\, x_2 \right)$

##### Constraints

1. Inventory Constraints:
\[
\begin{align*}
x_1 &\leq 62440 \\
x_2 &\leq 75500 \\
\end{align*}
\]

2. Demand Constraints:
\[
\begin{align*}
x_1 &\leq 12474 \\
x_2 &\leq 15057 \\
\end{align*}
\]

3. Non-negativity and Integrality:
\[
x_1, x_2 \geq 0,\quad x_1, x_2 \in \mathbb{Z}
\]

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "27in 4K Gaming Monitor",
      "Revenue": 261.2933,
      "Demand": 12474,
      "Initial Inventory": 62440
    },
    {
      "Product Name": "27in FHD Monitor",
      "Revenue": 52.4965,
      "Demand": 15057,
      "Initial Inventory": 75500
    }
  ]
}

##### Variable Index Mapping

- $x_1$: 27in 4K Gaming Monitor
- $x_2$: 27in FHD Monitor