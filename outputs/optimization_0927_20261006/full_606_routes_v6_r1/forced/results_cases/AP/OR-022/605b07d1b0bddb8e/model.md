##### Decision Variables:
Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘27in’ products) to be fulfilled.

##### Objective Function:
$\max \left( 261.2933\, x_1 + 52.4965\, x_2 \right)$

##### Constraints:
1. Inventory and Demand Fulfillment:
\[
\begin{align*}
0 \leq x_1 \leq \min(62440, 12474) \\
0 \leq x_2 \leq \min(75500, 15057)
\end{align*}
\]
That is,
\[
\begin{align*}
0 \leq x_1 \leq 12474 \\
0 \leq x_2 \leq 15057
\end{align*}
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

##### Variable Domains:
$x_1, x_2 \in \mathbb{Z}_{\geq 0}$ (if only integer units can be fulfilled; otherwise, $x_i \geq 0$ for continuous fulfillment)