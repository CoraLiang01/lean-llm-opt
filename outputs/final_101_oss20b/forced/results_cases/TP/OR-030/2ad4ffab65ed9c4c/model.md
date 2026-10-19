##### Decision Variables

$x_i\geq0$: quantity of car model $i$ (where $i\in I$) that the dealership plans to sell (continuous).

##### Objective Function

$\max\sum_{i\in I}r_i\,x_i$

##### Constraints

1. Demand: $\displaystyle x_i\leq d_i,\quad\forall i\in I$  
2. Initial inventory: $\displaystyle x_i\leq s_i,\quad\forall i\in I$  
3. Non‑negativity: $\displaystyle x_i\geq0,\quad\forall i\in I$

$ I=\{1,2,3,4,5,6,7,8\}$, where each index $i$ corresponds to a distinct ‘FDK57’ car model record in the order shown below.

###### Retrieved Information

```json
{
  "models": [
    {
      "index": 1,
      "Product Name": "FDK57",
      "Revenue": 119.144,
      "Demand": 30,
      "Initial Inventory": 200
    },
    {
      "index": 2,
      "Product Name": "FDK57",
      "Revenue": 119.144,
      "Demand": 40,
      "Initial Inventory": 100
    },
    {
      "index": 3,
      "Product Name": "FDK57",
      "Revenue": 121.244,
      "Demand": 30,
      "Initial Inventory": 200
    },
    {
      "index": 4,
      "Product Name": "FDK57",
      "Revenue": 120.144,
      "Demand": 50,
      "Initial Inventory": 150
    },
    {
      "index": 5,
      "Product Name": "FDK57",
      "Revenue": 120.544,
      "Demand": 10,
      "Initial Inventory": 150
    },
    {
      "index": 6,
      "Product Name": "FDK57",
      "Revenue": 121.244,
      "Demand": 30,
      "Initial Inventory": 150
    },
    {
      "index": 7,
      "Product Name": "FDK57",
      "Revenue": 119.744,
      "Demand": 50,
      "Initial Inventory": 250
    },
    {
      "index": 8,
      "Product Name": "FDK57",
      "Revenue": 120.844,
      "Demand": 50,
      "Initial Inventory": 150
    }
  ]
}
```