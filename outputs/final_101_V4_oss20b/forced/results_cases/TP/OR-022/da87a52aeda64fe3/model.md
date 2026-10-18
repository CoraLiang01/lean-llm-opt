##### Decision Variables

$x_i\geq0$: number of units of 27in product $i$ that will be fulfilled (continuous).

##### Objective Function

$\max\sum_{i\in I}r_i\,x_i$

##### Constraints

1. Demand: $\displaystyle x_i\leq d_i,\quad\forall i\in I$  
2. Initial inventory: $\displaystyle x_i\leq s_i,\quad\forall i\in I$  
3. Non‑negativity: $x_i\geq0,\quad\forall i\in I$

$I=\{1,2\}$, where product $1$ is “27in 4K Gaming Monitor” and product $2$ is “27in FHD Monitor”.

###### Retrieved Information

```json
{
  "products": [
    {
      "index": 1,
      "name": "27in 4K Gaming Monitor",
      "revenue": 261.2933,
      "demand": 12474,
      "initial_inventory": 62440
    },
    {
      "index": 2,
      "name": "27in FHD Monitor",
      "revenue": 52.4965,
      "demand": 15057,
      "initial_inventory": 75500
    }
  ],
  "revenue": {
    "1": 261.2933,
    "2": 52.4965
  },
  "demand": {
    "1": 12474,
    "2": 15057
  },
  "initial_inventory": {
    "1": 62440,
    "2": 75500
  }
}
```