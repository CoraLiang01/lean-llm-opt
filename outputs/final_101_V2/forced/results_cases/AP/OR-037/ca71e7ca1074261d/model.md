##### Objective Function:

$\quad \max \sum_{i=1}^{25} v_i x_i$

where $x_i$ is the number of vehicles of type $i$ to order per day, and $v_i$ is the profit per unit for vehicle type $i$.

##### Constraints

###### 1. Inventory Capacity Constraint:

$\sum_{i=1}^{25} w_i x_i \leq 765$

where $w_i$ is the weight (inventory space required) per unit for vehicle type $i$.

###### 2. Non-negativity and Integrality Constraints:

$x_i \geq 0$, integer, for all $i = 1, \ldots, 25$

##### Retrieved Information

{
  "capacity": 765,
  "products": [
    {"ProductName": "Sedan", "Value": 2524, "Weight": 99},
    {"ProductName": "SUV", "Value": 4614, "Weight": 55},
    {"ProductName": "Truck", "Value": 8416, "Weight": 75},
    {"ProductName": "Convertible", "Value": 5917, "Weight": 94},
    {"ProductName": "Minivan", "Value": 9048, "Weight": 80},
    {"ProductName": "Coupe", "Value": 1140, "Weight": 82},
    {"ProductName": "Hatchback", "Value": 8962, "Weight": 71},
    {"ProductName": "Station Wagon", "Value": 1888, "Weight": 100},
    {"ProductName": "Electric Car", "Value": 8487, "Weight": 28},
    {"ProductName": "Hybrid Car", "Value": 4425, "Weight": 93},
    {"ProductName": "Luxury Sedan", "Value": 4717, "Weight": 84},
    {"ProductName": "Sports Car", "Value": 4210, "Weight": 83},
    {"ProductName": "Crossover", "Value": 1226, "Weight": 62},
    {"ProductName": "Diesel Truck", "Value": 7400, "Weight": 90},
    {"ProductName": "Compact SUV", "Value": 4639, "Weight": 99},
    {"ProductName": "Luxury SUV", "Value": 7712, "Weight": 96},
    {"ProductName": "Cargo Van", "Value": 3299, "Weight": 21},
    {"ProductName": "Pickup Truck", "Value": 9895, "Weight": 39},
    {"ProductName": "Roadster", "Value": 4496, "Weight": 99},
    {"ProductName": "Muscle Car", "Value": 4526, "Weight": 81},
    {"ProductName": "Off-road Vehicle", "Value": 5688, "Weight": 6},
    {"ProductName": "Camper Van", "Value": 3007, "Weight": 58},
    {"ProductName": "Compact Car", "Value": 3623, "Weight": 37},
    {"ProductName": "Motorcycle", "Value": 8474, "Weight": 15},
    {"ProductName": "Electric SUV", "Value": 8372, "Weight": 37}
  ]
}

##### Full Model (with explicit parameters):

Let the vehicle types be indexed as follows:

1. Sedan: $v_1 = 2524$, $w_1 = 99$
2. SUV: $v_2 = 4614$, $w_2 = 55$
3. Truck: $v_3 = 8416$, $w_3 = 75$
4. Convertible: $v_4 = 5917$, $w_4 = 94$
5. Minivan: $v_5 = 9048$, $w_5 = 80$
6. Coupe: $v_6 = 1140$, $w_6 = 82$
7. Hatchback: $v_7 = 8962$, $w_7 = 71$
8. Station Wagon: $v_8 = 1888$, $w_8 = 100$
9. Electric Car: $v_9 = 8487$, $w_9 = 28$
10. Hybrid Car: $v_{10} = 4425$, $w_{10} = 93$
11. Luxury Sedan: $v_{11} = 4717$, $w_{11} = 84$
12. Sports Car: $v_{12} = 4210$, $w_{12} = 83$
13. Crossover: $v_{13} = 1226$, $w_{13} = 62$
14. Diesel Truck: $v_{14} = 7400$, $w_{14} = 90$
15. Compact SUV: $v_{15} = 4639$, $w_{15} = 99$
16. Luxury SUV: $v_{16} = 7712$, $w_{16} = 96$
17. Cargo Van: $v_{17} = 3299$, $w_{17} = 21$
18. Pickup Truck: $v_{18} = 9895$, $w_{18} = 39$
19. Roadster: $v_{19} = 4496$, $w_{19} = 99$
20. Muscle Car: $v_{20} = 4526$, $w_{20} = 81$
21. Off-road Vehicle: $v_{21} = 5688$, $w_{21} = 6$
22. Camper Van: $v_{22} = 3007$, $w_{22} = 58$
23. Compact Car: $v_{23} = 3623$, $w_{23} = 37$
24. Motorcycle: $v_{24} = 8474$, $w_{24} = 15$
25. Electric SUV: $v_{25} = 8372$, $w_{25} = 37$

$\max \left(2524x_1 + 4614x_2 + 8416x_3 + 5917x_4 + 9048x_5 + 1140x_6 + 8962x_7 + 1888x_8 + 8487x_9 + 4425x_{10} + 4717x_{11} + 4210x_{12} + 1226x_{13} + 7400x_{14} + 4639x_{15} + 7712x_{16} + 3299x_{17} + 9895x_{18} + 4496x_{19} + 4526x_{20} + 5688x_{21} + 3007x_{22} + 3623x_{23} + 8474x_{24} + 8372x_{25}\right)$

subject to

$99x_1 + 55x_2 + 75x_3 + 94x_4 + 80x_5 + 82x_6 + 71x_7 + 100x_8 + 28x_9 + 93x_{10} + 84x_{11} + 83x_{12} + 62x_{13} + 90x_{14} + 99x_{15} + 96x_{16} + 21x_{17} + 39x_{18} + 99x_{19} + 81x_{20} + 6x_{21} + 58x_{22} + 37x_{23} + 15x_{24} + 37x_{25} \leq 765$

$x_i \geq 0$, integer, for $i = 1, \ldots, 25$