##### Decision Variables

$x_i\geq0$: quantity of product $i$ fulfilled (continuous).

##### Objective Function

$\max\sum_{i}r_i\,x_i$

##### Constraints

1. Demand: $x_i\leq d_i,\quad\forall i$  
2. Inventory: $x_i\leq s_i,\quad\forall i$  
3. Non‑negativity: $x_i\geq0,\quad\forall i$

**Numerical instantiation**

1. Demand constraints  
   $x_1\leq 1980$  
   $x_2\leq 3024$  
   $x_3\leq 4536$  
   $x_4\leq 5601$  
   $x_5\leq 7567$

2. Inventory constraints  
   $x_1\leq 9920.0$  
   $x_2\leq 20160.0$  
   $x_3\leq 30000.0$  
   $x_4\leq 38360.0$  
   $x_5\leq 51450.0$

3. Non‑negativity  
   $x_i\geq0,\quad i=1,\dots,5$

4. Objective  
   $\max\;15.15\,x_1+30.3\,x_2+45.45\,x_3+60.6\,x_4+75.75\,x_5$

###### Retrieved Information

{
  "products": [
    {
      "name": "Books_15.15",
      "demand": 1980,
      "initial_inventory": 9920.0,
      "revenue": 15.15
    },
    {
      "name": "Books_30.3",
      "demand": 3024,
      "initial_inventory": 20160.0,
      "revenue": 30.3
    },
    {
      "name": "Books_45.45",
      "demand": 4536,
      "initial_inventory": 30000.0,
      "revenue": 45.45
    },
    {
      "name": "Books_60.6",
      "demand": 5601,
      "initial_inventory": 38360.0,
      "revenue": 60.6
    },
    {
      "name": "Books_75.75",
      "demand": 7567,
      "initial_inventory": 51450.0,
      "revenue": 75.75
    }
  ]
}