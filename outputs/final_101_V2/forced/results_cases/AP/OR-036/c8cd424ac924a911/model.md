##### Objective Function:

$\quad \max \sum_{i=1}^{25} v_i x_i$

where $x_i$ is the number of units of vehicle type $i$ to be ordered daily, and $v_i$ is the benefit coefficient for vehicle type $i$.

##### Constraints:

$\sum_{i=1}^{25} w_i x_i \leq 1576$

$x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,2,\ldots,25\}$

##### Retrieved Information

{
  "capacity": 1576,
  "products": [
    {"ProductName": "Sedan", "Value": 1752, "Weight": 15},
    {"ProductName": "SUV", "Value": 1856, "Weight": 87},
    {"ProductName": "Truck", "Value": 8372, "Weight": 36},
    {"ProductName": "Convertible", "Value": 6168, "Weight": 30},
    {"ProductName": "Minivan", "Value": 9681, "Weight": 33},
    {"ProductName": "Coupe", "Value": 8062, "Weight": 72},
    {"ProductName": "Hatchback", "Value": 3895, "Weight": 75},
    {"ProductName": "Station Wagon", "Value": 3254, "Weight": 71},
    {"ProductName": "Electric Car", "Value": 1701, "Weight": 51},
    {"ProductName": "Hybrid Car", "Value": 6799, "Weight": 21},
    {"ProductName": "Luxury Sedan", "Value": 2724, "Weight": 97},
    {"ProductName": "Sports Car", "Value": 6304, "Weight": 52},
    {"ProductName": "Crossover", "Value": 3255, "Weight": 25},
    {"ProductName": "Diesel Truck", "Value": 1923, "Weight": 15},
    {"ProductName": "Compact SUV", "Value": 4103, "Weight": 54},
    {"ProductName": "Luxury SUV", "Value": 4429, "Weight": 57},
    {"ProductName": "Cargo Van", "Value": 2663, "Weight": 18},
    {"ProductName": "Pickup Truck", "Value": 1691, "Weight": 69},
    {"ProductName": "Roadster", "Value": 5632, "Weight": 26},
    {"ProductName": "Muscle Car", "Value": 4793, "Weight": 38},
    {"ProductName": "Off-road Vehicle", "Value": 1343, "Weight": 31},
    {"ProductName": "Camper Van", "Value": 9124, "Weight": 74},
    {"ProductName": "Compact Car", "Value": 3652, "Weight": 82},
    {"ProductName": "Motorcycle", "Value": 8842, "Weight": 49},
    {"ProductName": "Electric SUV", "Value": 9176, "Weight": 64}
  ]
}

##### Parameter Vectors

Let the vehicle types be indexed as follows:

1. Sedan
2. SUV
3. Truck
4. Convertible
5. Minivan
6. Coupe
7. Hatchback
8. Station Wagon
9. Electric Car
10. Hybrid Car
11. Luxury Sedan
12. Sports Car
13. Crossover
14. Diesel Truck
15. Compact SUV
16. Luxury SUV
17. Cargo Van
18. Pickup Truck
19. Roadster
20. Muscle Car
21. Off-road Vehicle
22. Camper Van
23. Compact Car
24. Motorcycle
25. Electric SUV

Benefit coefficients ($v_i$):

$(1752,\ 1856,\ 8372,\ 6168,\ 9681,\ 8062,\ 3895,\ 3254,\ 1701,\ 6799,\ 2724,\ 6304,\ 3255,\ 1923,\ 4103,\ 4429,\ 2663,\ 1691,\ 5632,\ 4793,\ 1343,\ 9124,\ 3652,\ 8842,\ 9176)$

Weight coefficients ($w_i$):

$(15,\ 87,\ 36,\ 30,\ 33,\ 72,\ 75,\ 71,\ 51,\ 21,\ 97,\ 52,\ 25,\ 15,\ 54,\ 57,\ 18,\ 69,\ 26,\ 38,\ 31,\ 74,\ 82,\ 49,\ 64)$

##### Complete Mathematical Model

$\max \left(1752x_1 + 1856x_2 + 8372x_3 + 6168x_4 + 9681x_5 + 8062x_6 + 3895x_7 + 3254x_8 + 1701x_9 + 6799x_{10} + 2724x_{11} + 6304x_{12} + 3255x_{13} + 1923x_{14} + 4103x_{15} + 4429x_{16} + 2663x_{17} + 1691x_{18} + 5632x_{19} + 4793x_{20} + 1343x_{21} + 9124x_{22} + 3652x_{23} + 8842x_{24} + 9176x_{25}\right)$

subject to

$15x_1 + 87x_2 + 36x_3 + 30x_4 + 33x_5 + 72x_6 + 75x_7 + 71x_8 + 51x_9 + 21x_{10} + 97x_{11} + 52x_{12} + 25x_{13} + 15x_{14} + 54x_{15} + 57x_{16} + 18x_{17} + 69x_{18} + 26x_{19} + 38x_{20} + 31x_{21} + 74x_{22} + 82x_{23} + 49x_{24} + 64x_{25} \leq 1576$

$x_i \in \mathbb{Z}_{\geq 0}, \quad i = 1,2,\ldots,25$