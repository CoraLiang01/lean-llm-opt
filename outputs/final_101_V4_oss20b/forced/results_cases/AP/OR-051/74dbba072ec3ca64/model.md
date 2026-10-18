##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{10} \sum_{j=1}^{18} v_{j}\,x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$\sum_{j=1}^{18} w_{j}\,x_{ij} \le C_{i} \quad \forall i \in \{1,2,\dots ,10\}$

###### 2. Variable Constraints: 

$x_{ij} \in \mathbb{Z}_{\ge 0} \quad \forall i,j$

##### Retrieved Information

{
  "capacity": {
    "1": 400,
    "2": 600,
    "3": 500,
    "4": 700,
    "5": 450,
    "6": 650,
    "7": 550,
    "8": 750,
    "9": 480,
    "10": 520
  },
  "products": {
    "Espresso Beans": {"Value": 100, "Weight": 1.0},
    "Colombian Roast": {"Value": 150, "Weight": 1.5},
    "Arabica Blend": {"Value": 80, "Weight": 1.2},
    "French Roast": {"Value": 120, "Weight": 1.3},
    "Italian Roast": {"Value": 130, "Weight": 1.4},
    "House Blend": {"Value": 110, "Weight": 1.1},
    "Sumatra Coffee": {"Value": 160, "Weight": 1.8},
    "Mocha Java": {"Value": 90, "Weight": 1.2},
    "Hazelnut Flavor": {"Value": 95, "Weight": 1.0},
    "Caramel Blend": {"Value": 105, "Weight": 1.3},
    "Vanilla Flavor": {"Value": 85, "Weight": 1.2},
    "Cappuccino Mix": {"Value": 140, "Weight": 1.5},
    "Pumpkin Spice": {"Value": 75, "Weight": 1.1},
    "Decaf Roast": {"Value": 60, "Weight": 1.0},
    "Organic Roast": {"Value": 170, "Weight": 1.6},
    "Cold Brew": {"Value": 115, "Weight": 1.4},
    "Peruvian Blend": {"Value": 155, "Weight": 1.7},
    "Kenyan AA": {"Value": 125, "Weight": 1.3}
  },
  "cabinets": ["1","2","3","4","5","6","7","8","9","10"],
  "products_list": [
    "Espresso Beans",
    "Colombian Roast",
    "Arabica Blend",
    "French Roast",
    "Italian Roast",
    "House Blend",
    "Sumatra Coffee",
    "Mocha Java",
    "Hazelnut Flavor",
    "Caramel Blend",
    "Vanilla Flavor",
    "Cappuccino Mix",
    "Pumpkin Spice",
    "Decaf Roast",
    "Organic Roast",
    "Cold Brew",
    "Peruvian Blend",
    "Kenyan AA"
  ]
}