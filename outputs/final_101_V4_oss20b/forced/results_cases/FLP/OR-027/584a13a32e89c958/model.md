##### Decision Variables

$x_i \geq 0$: number of units of product $i$ fulfilled.

##### Objective Function

$\max \sum_{i} r_i x_i$

##### Constraints

1. Demand constraint: $x_i \leq d_i,\quad \forall i$.

2. Inventory constraint: $x_i \leq I_i,\quad \forall i$.

3. Domain: $x_i \geq 0$.

###### Retrieved Information

{
  "products": [
    "Organic Fruits",
    "Organic Staples",
    "Organic Vegetables"
  ],
  "demand": {
    "Organic Fruits": 678906,
    "Organic Staples": 749927,
    "Organic Vegetables": 699808
  },
  "inventory": {
    "Organic Fruits": 5034020.0,
    "Organic Staples": 5589290.0,
    "Organic Vegetables": 5202710.0
  },
  "revenue": {
    "Organic Fruits": 60.8,
    "Organic Staples": 918.45,
    "Organic Vegetables": 77.52
  }
}