##### Objective Function:

$\quad \max \sum_{i \in \text{Aalop}} r_i x_i$

where $r_i$ is the revenue per unit for product $i$, and $x_i$ is the number of units of product $i$ to fulfill.

##### Constraints:

For each product $i$ classified as 'Aalop':

$0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

##### Variable Domains:

$x_i \geq 0$ and integer, for all $i$ in 'Aalop' products.

##### Retrieved Information

{
  "Aalop": [
    {
      "Product Name": "Aalopuri",
      "Revenue": 20,
      "Demand": 1483,
      "Initial Inventory": 10440.0
    }
  ]
}

##### Model Parameters

- Product set: $\text{Aalop} = \{\text{Aalopuri}\}$
- $r_{\text{Aalopuri}} = 20$
- $\text{Demand}_{\text{Aalopuri}} = 1483$
- $\text{Initial Inventory}_{\text{Aalopuri}} = 10440.0$

##### Complete Model

$\max\ 20\, x_{\text{Aalopuri}}$

subject to

$0 \leq x_{\text{Aalopuri}} \leq 1483$

$x_{\text{Aalopuri}}$ integer