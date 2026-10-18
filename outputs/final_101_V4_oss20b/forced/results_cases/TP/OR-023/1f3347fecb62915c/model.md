##### Decision Variables

$x_i\geq0$: number of units of product $i$ that the retailer plans to fulfill, for each $i\in I$ (continuous).

##### Objective Function

$\max\sum_{i\in I}r_i\,x_i$

##### Constraints

1. Demand: $\displaystyle x_i\leq d_i,\quad\forall i\in I$  
2. Initial inventory: $\displaystyle x_i\leq s_i,\quad\forall i\in I$  
3. Non‑negativity: $x_i\geq0,\quad\forall i\in I$

$I=\{\,\text{ELE-SMA-10000463},\text{ELE-SMA-10000487},\text{ELE-SMA-10003333},\text{ELE-SMA-10009012},\text{ELE-SMA-10009999},\text{ELE-SMA-10011234},\text{ELE-SMA-10027456},\text{ELE-SMA-10028567},\text{ELE-SPE-10000484},\text{ELE-SPE-10003030},\text{ELE-SPE-10024123},\text{ELE-SPE-10025234}\,\}$  

$d_i$ denotes the demand for product $i$, $s_i$ denotes the initial inventory for product $i$, and $r_i$ denotes the revenue per unit for product $i$.

###### Retrieved Information

{
  "products": [
    {"Product_Reference":"ELE-SMA-10000463","Revenue":4.0,"Demand":295,"Initial_Inventory":2000.0},
    {"Product_Reference":"ELE-SMA-10000487","Revenue":14.0,"Demand":1002,"Initial_Inventory":7000.0},
    {"Product_Reference":"ELE-SMA-10003333","Revenue":14.0,"Demand":958,"Initial_Inventory":7000.0},
    {"Product_Reference":"ELE-SMA-10009012","Revenue":4.0,"Demand":777,"Initial_Inventory":6000.0},
    {"Product_Reference":"ELE-SMA-10009999","Revenue":4.0,"Demand":271,"Initial_Inventory":2000.0},
    {"Product_Reference":"ELE-SMA-10011234","Revenue":4.0,"Demand":244,"Initial_Inventory":2000.0},
    {"Product_Reference":"ELE-SMA-10027456","Revenue":14.0,"Demand":990,"Initial_Inventory":7000.0},
    {"Product_Reference":"ELE-SMA-10028567","Revenue":14.0,"Demand":1000,"Initial_Inventory":7000.0},
    {"Product_Reference":"ELE-SPE-10000484","Revenue":2.4,"Demand":169,"Initial_Inventory":1200.0},
    {"Product_Reference":"ELE-SPE-10003030","Revenue":2.4,"Demand":155,"Initial_Inventory":1200.0},
    {"Product_Reference":"ELE-SPE-10024123","Revenue":2.4,"Demand":327,"Initial_Inventory":2400.0},
    {"Product_Reference":"ELE-SPE-10025234","Revenue":2.4,"Demand":174,"Initial_Inventory":1200.0}
  ]
}