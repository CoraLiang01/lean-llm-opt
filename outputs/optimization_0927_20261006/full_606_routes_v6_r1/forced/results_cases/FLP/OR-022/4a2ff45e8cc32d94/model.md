##### Decision Variables

$x_i \geq 0$: Number of units of product $i$ (where $i$ is a 27in product) to fulfill (integer, $i \in I$).

##### Parameters

Let $I = \{$"27in 4K Gaming Monitor", "27in FHD Monitor"$\}$

- Revenue per unit:
  - $r_{\text{27in 4K Gaming Monitor}} = 261.2933$
  - $r_{\text{27in FHD Monitor}} = 52.4965$
- Initial Inventory:
  - $s_{\text{27in 4K Gaming Monitor}} = 62440$
  - $s_{\text{27in FHD Monitor}} = 75500$
- Demand:
  - $d_{\text{27in 4K Gaming Monitor}} = 12474$
  - $d_{\text{27in FHD Monitor}} = 15057$

##### Objective Function

\[
\max \left( 261.2933\, x_{\text{27in 4K Gaming Monitor}} + 52.4965\, x_{\text{27in FHD Monitor}} \right)
\]

##### Constraints

1. Inventory and demand limits for each product:
   \[
   0 \leq x_{\text{27in 4K Gaming Monitor}} \leq \min(62440, 12474) = 12474
   \]
   \[
   0 \leq x_{\text{27in FHD Monitor}} \leq \min(75500, 15057) = 15057
   \]

2. $x_i$ are integer variables (if partial units are not allowed; otherwise, continuous and nonnegative).

##### Complete Model

\[
\begin{align*}
\max\quad & 261.2933\, x_{\text{27in 4K Gaming Monitor}} + 52.4965\, x_{\text{27in FHD Monitor}} \\
\text{s.t.}\quad & 0 \leq x_{\text{27in 4K Gaming Monitor}} \leq 12474 \\
                 & 0 \leq x_{\text{27in FHD Monitor}} \leq 15057 \\
                 & x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} \geq 0 \\
                 & x_{\text{27in 4K Gaming Monitor}},\ x_{\text{27in FHD Monitor}} \in \mathbb{Z} \quad \text{(if integer units required)}
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