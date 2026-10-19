##### Decision Variables

$x_i\geq0$: number of units of product $i$ that will be fulfilled, for each $i\in I$ (continuous).

##### Objective Function

$\max\sum_{i\in I}r_i\,x_i$

##### Constraints

1. Inventory capacity: $\displaystyle x_i\leq s_i,\quad\forall i\in I$  
2. Demand satisfaction: $\displaystyle x_i\leq d_i,\quad\forall i\in I$  
3. Non‑negativity: $x_i\geq0,\quad\forall i\in I$

$I=\{1,2,\dots,12\}$, where each index $i$ corresponds to a distinct ‘FAUX’ product in the order listed below.

###### Retrieved Information

{
  "products": [
    {
      "Product Name": "FAUX FUR JEWEL SWEATER",
      "Revenue": 35.9,
      "Initial Inventory": 20970,
      "Demand": 3025
    },
    {
      "Product Name": "FAUX LEATHER BOMBER JACKET",
      "Revenue": 69.9,
      "Initial Inventory": 71970,
      "Demand": 9585
    },
    {
      "Product Name": "FAUX LEATHER BOXY FIT JACKET",
      "Revenue": 99.9,
      "Initial Inventory": 32730,
      "Demand": 4486
    },
    {
      "Product Name": "FAUX LEATHER JACKET",
      "Revenue": 99.9,
      "Initial Inventory": 71130,
      "Demand": 10322
    },
    {
      "Product Name": "FAUX LEATHER OVERSIZED JACKET LIMITED EDITION",
      "Revenue": 159.0,
      "Initial Inventory": 34910,
      "Demand": 4868
    },
    {
      "Product Name": "FAUX LEATHER PUFFER JACKET",
      "Revenue": 69.99,
      "Initial Inventory": 64010,
      "Demand": 8482
    },
    {
      "Product Name": "FAUX SHEARLING LINED SUEDE BOOTS",
      "Revenue": 99.9,
      "Initial Inventory": 20760,
      "Demand": 2607
    },
    {
      "Product Name": "FAUX SHEARLING PLAID JACKET",
      "Revenue": 89.9,
      "Initial Inventory": 12490,
      "Demand": 1784
    },
    {
      "Product Name": "FAUX SUEDE BOMBER JACKET",
      "Revenue": 69.9,
      "Initial Inventory": 50300,
      "Demand": 6626
    },
    {
      "Product Name": "FAUX SUEDE JACKET",
      "Revenue": 89.9,
      "Initial Inventory": 24570,
      "Demand": 3256
    },
    {
      "Product Name": "FAUX SUEDE OVERSHIRT",
      "Revenue": 69.9,
      "Initial Inventory": 24430,
      "Demand": 2955
    },
    {
      "Product Name": "FAUX SUEDE PATCH JACKET",
      "Revenue": 89.9,
      "Initial Inventory": 7070,
      "Demand": 910
    }
  ],
  "variables": {
    "x_i": "continuous, nonnegative"
  },
  "objective": {
    "sense": "maximize",
    "expression": "sum over i of Revenue_i * x_i"
  },
  "constraints": [
    {
      "type": "upper bound",
      "expression": "x_i <= Initial Inventory_i"
    },
    {
      "type": "upper bound",
      "expression": "x_i <= Demand_i"
    },
    {
      "type": "nonnegativity",
      "expression": "x_i >= 0"
    }
  ]
}