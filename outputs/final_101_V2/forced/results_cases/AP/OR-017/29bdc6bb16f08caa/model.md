##### Decision Variables

Let $x_i$ denote the number of units of product $i$ (where $i$ is a SKU in the ZZ category) to fulfill.

##### Objective Function

$\quad \max \sum_{i \in \{\text{ZZ2AO},\, \text{ZZD W7},\, \text{ZZM1A},\, \text{ZZNC5},\, \text{ZZX6K}\}} \text{Revenue}_i \cdot x_i$

That is,

$\quad \max \left(24.38\, x_{\text{ZZ2AO}} + 30.12\, x_{\text{ZZD W7}} + 19.52\, x_{\text{ZZM1A}} + 10.79\, x_{\text{ZZNC5}} + 111.81\, x_{\text{ZZX6K}}\right)$

##### Constraints

1. Inventory constraints:
\[
\begin{align*}
x_{\text{ZZ2AO}} &\leq 10.0 \\
x_{\text{ZZD W7}} &\leq 20.0 \\
x_{\text{ZZM1A}} &\leq 530.0 \\
x_{\text{ZZNC5}} &\leq 10.0 \\
x_{\text{ZZX6K}} &\leq 10.0 \\
\end{align*}
\]

2. Demand constraints:
\[
\begin{align*}
x_{\text{ZZ2AO}} &\leq 2 \\
x_{\text{ZZD W7}} &\leq 4 \\
x_{\text{ZZM1A}} &\leq 82 \\
x_{\text{ZZNC5}} &\leq 2 \\
x_{\text{ZZX6K}} &\leq 2 \\
\end{align*}
\]

3. Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{\text{ZZ2AO},\, \text{ZZD W7},\, \text{ZZM1A},\, \text{ZZNC5},\, \text{ZZX6K}\}
\]

##### Retrieved Information

{
  "products": [
    {
      "SKU": "ZZ2AO",
      "Revenue": 24.38,
      "Initial Inventory": 10.0,
      "Demand": 2
    },
    {
      "SKU": "ZZD W7",
      "Revenue": 30.12,
      "Initial Inventory": 20.0,
      "Demand": 4
    },
    {
      "SKU": "ZZM1A",
      "Revenue": 19.52,
      "Initial Inventory": 530.0,
      "Demand": 82
    },
    {
      "SKU": "ZZNC5",
      "Revenue": 10.79,
      "Initial Inventory": 10.0,
      "Demand": 2
    },
    {
      "SKU": "ZZX6K",
      "Revenue": 111.81,
      "Initial Inventory": 10.0,
      "Demand": 2
    }
  ]
}