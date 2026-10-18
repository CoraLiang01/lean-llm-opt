##### Decision Variables

$x_i \ge 0$: quantity of product $i$ to be fulfilled.

##### Objective Function

$\max \sum_{i} rev_i \, x_i$

##### Constraints

1. Demand: $\forall i:\; x_i \le demand_i$  
2. Inventory: $\forall i:\; x_i \le initial\_inventory_i$  
3. Non‑negativity: $\forall i:\; x_i \ge 0$

###### Retrieved Information

{
  "products": [
    {
      "name": "Baby Food_255.28",
      "revenue": 255.28,
      "demand": 3066513,
      "initial_inventory": 22749210
    }
  ]
}