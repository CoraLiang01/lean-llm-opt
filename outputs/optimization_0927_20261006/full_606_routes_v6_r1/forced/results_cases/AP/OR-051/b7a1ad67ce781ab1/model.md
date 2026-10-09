##### Objective Function:

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{17} v_j \, x_{ij}$

where $x_{ij}$ is the integer number of units of product $j$ placed in cabinet $i$, and $v_j$ is the value per unit of product $j$.

##### Constraints

###### 1. Cabinet Capacity Constraints:

$\sum_{j=1}^{17} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

where $w_j$ is the weight per unit of product $j$, and $C_i$ is the capacity of cabinet $i$.

###### 2. Variable Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,10\}, \; j \in \{1,2,\ldots,17\}$

##### Retrieved Information

{
  "cabinets": [
    {"CabinetID": 1, "Capacity": 400},
    {"CabinetID": 2, "Capacity": 600},
    {"CabinetID": 3, "Capacity": 500},
    {"CabinetID": 4, "Capacity": 700},
    {"CabinetID": 5, "Capacity": 450},
    {"CabinetID": 6, "Capacity": 650},
    {"CabinetID": 7, "Capacity": 550},
    {"CabinetID": 8, "Capacity": 750},
    {"CabinetID": 9, "Capacity": 480},
    {"CabinetID": 10, "Capacity": 520}
  ],
  "products": [
    {"ProductName": "Espresso Beans", "Value": 100, "Weight": 1.0},
    {"ProductName": "Colombian Roast", "Value": 150, "Weight": 1.5},
    {"ProductName": "Arabica Blend", "Value": 80, "Weight": 1.2},
    {"ProductName": "French Roast", "Value": 120, "Weight": 1.3},
    {"ProductName": "Italian Roast", "Value": 130, "Weight": 1.4},
    {"ProductName": "House Blend", "Value": 110, "Weight": 1.1},
    {"ProductName": "Sumatra Coffee", "Value": 160, "Weight": 1.8},
    {"ProductName": "Mocha Java", "Value": 90, "Weight": 1.2},
    {"ProductName": "Hazelnut Flavor", "Value": 95, "Weight": 1.0},
    {"ProductName": "Caramel Blend", "Value": 105, "Weight": 1.3},
    {"ProductName": "Vanilla Flavor", "Value": 85, "Weight": 1.2},
    {"ProductName": "Cappuccino Mix", "Value": 140, "Weight": 1.5},
    {"ProductName": "Pumpkin Spice", "Value": 75, "Weight": 1.1},
    {"ProductName": "Decaf Roast", "Value": 60, "Weight": 1.0},
    {"ProductName": "Organic Roast", "Value": 170, "Weight": 1.6},
    {"ProductName": "Cold Brew", "Value": 115, "Weight": 1.4},
    {"ProductName": "Peruvian Blend", "Value": 155, "Weight": 1.7},
    {"ProductName": "Kenyan AA", "Value": 125, "Weight": 1.3}
  ]
}

##### Index Mapping

Let $i$ index cabinets as follows:
1: Cabinet 1 (Capacity 400)  
2: Cabinet 2 (Capacity 600)  
3: Cabinet 3 (Capacity 500)  
4: Cabinet 4 (Capacity 700)  
5: Cabinet 5 (Capacity 450)  
6: Cabinet 6 (Capacity 650)  
7: Cabinet 7 (Capacity 550)  
8: Cabinet 8 (Capacity 750)  
9: Cabinet 9 (Capacity 480)  
10: Cabinet 10 (Capacity 520)  

Let $j$ index products as follows:
1: Espresso Beans (Value 100, Weight 1.0)  
2: Colombian Roast (Value 150, Weight 1.5)  
3: Arabica Blend (Value 80, Weight 1.2)  
4: French Roast (Value 120, Weight 1.3)  
5: Italian Roast (Value 130, Weight 1.4)  
6: House Blend (Value 110, Weight 1.1)  
7: Sumatra Coffee (Value 160, Weight 1.8)  
8: Mocha Java (Value 90, Weight 1.2)  
9: Hazelnut Flavor (Value 95, Weight 1.0)  
10: Caramel Blend (Value 105, Weight 1.3)  
11: Vanilla Flavor (Value 85, Weight 1.2)  
12: Cappuccino Mix (Value 140, Weight 1.5)  
13: Pumpkin Spice (Value 75, Weight 1.1)  
14: Decaf Roast (Value 60, Weight 1.0)  
15: Organic Roast (Value 170, Weight 1.6)  
16: Cold Brew (Value 115, Weight 1.4)  
17: Peruvian Blend (Value 155, Weight 1.7)  
18: Kenyan AA (Value 125, Weight 1.3)  

##### Parameters

- $C_i$: Capacity of cabinet $i$ (see cabinets list above)
- $v_j$: Value per unit of product $j$ (see products list above)
- $w_j$: Weight per unit of product $j$ (see products list above)

##### Decision Variables

- $x_{ij}$: Integer number of units of product $j$ placed in cabinet $i$, $x_{ij} \geq 0$