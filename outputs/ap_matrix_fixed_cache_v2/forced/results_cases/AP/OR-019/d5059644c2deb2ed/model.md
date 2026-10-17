##### Decision Variables

Let $x_i$ denote the number of units of product $i$ (where $i$ indexes the set of ‘27in’ products) to fulfill.

##### Objective Function

$\max \sum_{i} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints

For each product $i$:
- $0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\}$

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "27in 4K Gaming Monitor",
      "Revenue": 389.99,
      "Demand": 12474,
      "Initial Inventory": 62440
    },
    {
      "Product Name": "27in FHD Monitor",
      "Revenue": 149.99,
      "Demand": 15057,
      "Initial Inventory": 75500
    }
  ]
}

##### Explicit Model

Let $x_1$ = units fulfilled of "27in 4K Gaming Monitor"  
Let $x_2$ = units fulfilled of "27in FHD Monitor"

$\max \ 389.99\, x_1 + 149.99\, x_2$

Subject to:
- $0 \leq x_1 \leq 12474$
- $0 \leq x_2 \leq 15057$

$x_1, x_2$ are integers (if only whole units can be fulfilled; otherwise, nonnegative real numbers if partial units are allowed).