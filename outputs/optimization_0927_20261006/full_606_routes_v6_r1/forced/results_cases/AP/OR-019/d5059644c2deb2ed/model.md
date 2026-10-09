##### Decision Variables:

Let $x_i$ denote the number of units of product $i$ (among ‘27in’ products) to fulfill, for each $i$ in the set of ‘27in’ products.

##### Objective Function:

$\quad \max \sum_{i} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints:

For each product $i$:

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

or, equivalently, separated:

$x_i \leq \text{Initial Inventory}_i$

$x_i \leq \text{Demand}_i$

$x_i \geq 0$

$x_i \in \mathbb{Z}$

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "27in 4K Gaming Monitor",
      "Revenue": 389.99,
      "Initial Inventory": 62440,
      "Demand": 12474
    },
    {
      "Product Name": "27in FHD Monitor",
      "Revenue": 149.99,
      "Initial Inventory": 75500,
      "Demand": 15057
    }
  ]
}

##### Model with Parameters

Let $x_1$ = units fulfilled of "27in 4K Gaming Monitor"

Let $x_2$ = units fulfilled of "27in FHD Monitor"

$\max \ 389.99\, x_1 + 149.99\, x_2$

subject to:

$\quad x_1 \leq 62440$

$\quad x_1 \leq 12474$

$\quad x_2 \leq 75500$

$\quad x_2 \leq 15057$

$\quad x_1 \geq 0,\ x_2 \geq 0$

$\quad x_1, x_2 \in \mathbb{Z}$

##### Variable and Parameter Summary

- $x_1$: units of "27in 4K Gaming Monitor" to fulfill
- $x_2$: units of "27in FHD Monitor" to fulfill
- $r_1 = 389.99$, $r_2 = 149.99$
- Initial Inventory: $62440$ (product 1), $75500$ (product 2)
- Demand: $12474$ (product 1), $15057$ (product 2)