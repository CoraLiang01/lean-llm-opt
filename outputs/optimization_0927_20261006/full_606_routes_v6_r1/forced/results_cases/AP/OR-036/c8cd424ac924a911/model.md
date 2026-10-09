##### Objective Function:

$\quad \max \sum_{i \in \mathcal{P}} v_i x_i$

where:
- $\mathcal{P}$ is the set of vehicle types (products)
- $v_i$ is the benefit coefficient for vehicle type $i$
- $x_i$ is the integer number of units of vehicle type $i$ to order daily

##### Constraints:

$\sum_{i \in \mathcal{P}} w_i x_i \leq C$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{P}$

where:
- $w_i$ is the weight (inventory usage per unit) for vehicle type $i$
- $C$ is the total inventory capacity

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

##### Full Model with Parameters

Let $\mathcal{P} = \{$Sedan, SUV, Truck, Convertible, Minivan, Coupe, Hatchback, Station Wagon, Electric Car, Hybrid Car, Luxury Sedan, Sports Car, Crossover, Diesel Truck, Compact SUV, Luxury SUV, Cargo Van, Pickup Truck, Roadster, Muscle Car, Off-road Vehicle, Camper Van, Compact Car, Motorcycle, Electric SUV$\}$.

Let $v_i$ and $w_i$ be as follows:

| ProductName         | $v_i$ | $w_i$ |
|---------------------|-------|-------|
| Sedan               | 1752  | 15    |
| SUV                 | 1856  | 87    |
| Truck               | 8372  | 36    |
| Convertible         | 6168  | 30    |
| Minivan             | 9681  | 33    |
| Coupe               | 8062  | 72    |
| Hatchback           | 3895  | 75    |
| Station Wagon       | 3254  | 71    |
| Electric Car        | 1701  | 51    |
| Hybrid Car          | 6799  | 21    |
| Luxury Sedan        | 2724  | 97    |
| Sports Car          | 6304  | 52    |
| Crossover           | 3255  | 25    |
| Diesel Truck        | 1923  | 15    |
| Compact SUV         | 4103  | 54    |
| Luxury SUV          | 4429  | 57    |
| Cargo Van           | 2663  | 18    |
| Pickup Truck        | 1691  | 69    |
| Roadster            | 5632  | 26    |
| Muscle Car          | 4793  | 38    |
| Off-road Vehicle    | 1343  | 31    |
| Camper Van          | 9124  | 74    |
| Compact Car         | 3652  | 82    |
| Motorcycle          | 8842  | 49    |
| Electric SUV        | 9176  | 64    |

Total inventory capacity: $C = 1576$

##### Complete Mathematical Model

$\max \Big($
$1752\,x_{\text{Sedan}} + 1856\,x_{\text{SUV}} + 8372\,x_{\text{Truck}} + 6168\,x_{\text{Convertible}} + 9681\,x_{\text{Minivan}} + 8062\,x_{\text{Coupe}} + 3895\,x_{\text{Hatchback}} + 3254\,x_{\text{Station Wagon}} + 1701\,x_{\text{Electric Car}} + 6799\,x_{\text{Hybrid Car}} + 2724\,x_{\text{Luxury Sedan}} + 6304\,x_{\text{Sports Car}} + 3255\,x_{\text{Crossover}} + 1923\,x_{\text{Diesel Truck}} + 4103\,x_{\text{Compact SUV}} + 4429\,x_{\text{Luxury SUV}} + 2663\,x_{\text{Cargo Van}} + 1691\,x_{\text{Pickup Truck}} + 5632\,x_{\text{Roadster}} + 4793\,x_{\text{Muscle Car}} + 1343\,x_{\text{Off-road Vehicle}} + 9124\,x_{\text{Camper Van}} + 3652\,x_{\text{Compact Car}} + 8842\,x_{\text{Motorcycle}} + 9176\,x_{\text{Electric SUV}}$
$\Big)$

subject to

$15\,x_{\text{Sedan}} + 87\,x_{\text{SUV}} + 36\,x_{\text{Truck}} + 30\,x_{\text{Convertible}} + 33\,x_{\text{Minivan}} + 72\,x_{\text{Coupe}} + 75\,x_{\text{Hatchback}} + 71\,x_{\text{Station Wagon}} + 51\,x_{\text{Electric Car}} + 21\,x_{\text{Hybrid Car}} + 97\,x_{\text{Luxury Sedan}} + 52\,x_{\text{Sports Car}} + 25\,x_{\text{Crossover}} + 15\,x_{\text{Diesel Truck}} + 54\,x_{\text{Compact SUV}} + 57\,x_{\text{Luxury SUV}} + 18\,x_{\text{Cargo Van}} + 69\,x_{\text{Pickup Truck}} + 26\,x_{\text{Roadster}} + 38\,x_{\text{Muscle Car}} + 31\,x_{\text{Off-road Vehicle}} + 74\,x_{\text{Camper Van}} + 82\,x_{\text{Compact Car}} + 49\,x_{\text{Motorcycle}} + 64\,x_{\text{Electric SUV}} \leq 1576$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{P}$