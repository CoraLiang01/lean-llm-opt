##### Decision Variables

$x_i\geq0$: quantity of product $i$ fulfilled, where $i\in I$.

##### Objective Function

$\max\sum_{i\in I}r_i x_i$

##### Constraints

1. Demand: $x_i\leq d_i,\quad\forall i\in I$.
2. Inventory: $x_i\leq s_i,\quad\forall i\in I$.
3. Non-negativity: $x_i\geq0$.

$I=\{\text{Organic Fruits},\text{Organic Staples},\text{Organic Vegetables}\}$.

###### Retrieved Information

{
  "products": [
    "Organic Fruits",
    "Organic Staples",
    "Organic Vegetables"
  ],
  "revenue": {
    "Organic Fruits": 60.8,
    "Organic Staples": 918.45,
    "Organic Vegetables": 77.52
  },
  "initial_inventory": {
    "Organic Fruits": 5034020.0,
    "Organic Staples": 5589290.0,
    "Organic Vegetables": 5202710.0
  },
  "demand": {
    "Organic Fruits": 678906,
    "Organic Staples": 749927,
    "Organic Vegetables": 699808
  }
}