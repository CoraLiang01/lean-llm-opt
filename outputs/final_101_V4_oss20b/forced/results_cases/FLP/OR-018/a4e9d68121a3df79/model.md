##### Decision Variables

$x_{1} \geq 0$: number of units of the Baby product $i=1$ that the retailer plans to fulfill (continuous).

##### Objective Function

$\max 255.28\,x_{1}$

##### Constraints

1. Demand satisfaction: $x_{1}\leq 3066513$  
2. Inventory limitation: $x_{1}\leq 22749210$  
3. Non‑negativity: $x_{1}\geq 0$  

##### Retrieved Information

{
  "products": [
    {
      "Product Name": "Baby Food_255.28",
      "Revenue": 255.28,
      "Demand": 3066513,
      "Initial Inventory": 22749210
    }
  ],
  "variables": {
    "x_1": "quantity of Baby Food_255.28 to sell"
  },
  "objective": {
    "coefficient": 255.28,
    "variable": "x_1"
  },
  "constraints": [
    {
      "type": "upper bound",
      "variable": "x_1",
      "value": 3066513,
      "description": "Demand limit"
    },
    {
      "type": "upper bound",
      "variable": "x_1",
      "value": 22749210,
      "description": "Initial inventory limit"
    },
    {
      "type": "nonnegativity",
      "variable": "x_1"
    }
  ]
}