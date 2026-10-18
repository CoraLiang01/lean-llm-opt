##### Decision Variables

$x_i \ge 0$: number of units of product $i$ to order each day (continuous).  

##### Objective Function

$\max \sum_{i=1}^{10} v_i x_i$

##### Constraints

1. Capacity constraint: $\sum_{i=1}^{10} w_i x_i \le C$  
2. Non‑negativity: $x_i \ge 0,\quad \forall i=1,\dots,10$  

##### Retrieved Information

{
  "products": [
    {"ProductName":"Spinach","Value":64,"Weight":230},
    {"ProductName":"Shiitake Mushrooms","Value":75,"Weight":637},
    {"ProductName":"Apples","Value":68,"Weight":773},
    {"ProductName":"Carrots","Value":11,"Weight":653},
    {"ProductName":"Basil","Value":91,"Weight":755},
    {"ProductName":"Potatoes","Value":31,"Weight":670},
    {"ProductName":"Green Beans","Value":90,"Weight":505},
    {"ProductName":"Blueberries","Value":56,"Weight":821},
    {"ProductName":"Oranges","Value":10,"Weight":83},
    {"ProductName":"Watermelons","Value":24,"Weight":249}
  ],
  "capacity": 875,
  "v": [64,75,68,11,91,31,90,56,10,24],
  "w": [230,637,773,653,755,670,505,821,83,249]
}