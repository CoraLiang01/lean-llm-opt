##### Decision Variables

$x_i \geq 0$ (integer): number of units of each ‘Baby’ product $i$ to fulfill.

##### Parameters

- Let $I$ be the set of ‘Baby’ products. Here, $I = \{\text{Baby Food\_255.28}\}$.
- For each $i \in I$:
  - $r_i$: Revenue per unit of product $i$.
  - $d_i$: Demand for product $i$.
  - $s_i$: Initial inventory of product $i$.

From the data:
- $r_{\text{Baby Food\_255.28}} = 255.28$
- $d_{\text{Baby Food\_255.28}} = 3,\!066,\!513$
- $s_{\text{Baby Food\_255.28}} = 22,\!749,\!210$

##### Objective Function

$\max \sum_{i \in I} r_i x_i$

##### Constraints

1. Demand fulfillment: $x_i \leq d_i,\quad \forall i \in I$
2. Inventory limit:   $x_i \leq s_i,\quad \forall i \in I$
3. Non-negativity and integrality: $x_i \geq 0$ and integer, $\forall i \in I$

##### Numerical Model

Let $x_{\text{Baby Food\_255.28}}$ be the number of units of Baby Food_255.28 to fulfill.

$\max\ 255.28\, x_{\text{Baby Food\_255.28}}$

subject to

$x_{\text{Baby Food\_255.28}} \leq 3,\!066,\!513$

$x_{\text{Baby Food\_255.28}} \leq 22,\!749,\!210$

$x_{\text{Baby Food\_255.28}} \geq 0$ and integer

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "Baby Food_255.28",
      "Revenue": 255.28,
      "Demand": 3066513,
      "Initial Inventory": 22749210
    }
  ]
}