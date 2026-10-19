##### Decision Variables

$x_{i}\geq0$: number of units of 27in product $i$ that the company plans to fulfill (continuous).

##### Objective Function

$\max 389.99\,x_{1}+149.99\,x_{2}$

##### Constraints

1. Demand: $x_{1}\leq12474,\quad x_{2}\leq15057$.
2. Initial inventory: $x_{1}\leq62440,\quad x_{2}\leq75500$.
3. Non‑negativity: $x_{i}\geq0,\quad i=1,2$.

###### Retrieved Information

{
  "products": [
    {
      "name": "27in 4K Gaming Monitor",
      "revenue": 389.99,
      "demand": 12474,
      "initial_inventory": 62440
    },
    {
      "name": "27in FHD Monitor",
      "revenue": 149.99,
      "demand": 15057,
      "initial_inventory": 75500
    }
  ],
  "variables": [
    {
      "name": "x_1",
      "description": "units of 27in 4K Gaming Monitor to fulfill"
    },
    {
      "name": "x_2",
      "description": "units of 27in FHD Monitor to fulfill"
    }
  ],
  "objective": {
    "sense": "maximize",
    "expression": "389.99*x_1 + 149.99*x_2"
  },
  "constraints": [
    {
      "name": "demand_1",
      "expression": "x_1 <= 12474"
    },
    {
      "name": "demand_2",
      "expression": "x_2 <= 15057"
    },
    {
      "name": "inventory_1",
      "expression": "x_1 <= 62440"
    },
    {
      "name": "inventory_2",
      "expression": "x_2 <= 75500"
    },
    {
      "name": "nonnegativity",
      "expression": "x_1 >= 0, x_2 >= 0"
    }
  ]
}