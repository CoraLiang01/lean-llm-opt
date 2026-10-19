##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: number of units of vehicle type $i$ to be ordered daily, for each $i \in I$.

##### Objective Function

$\max \sum_{i\in I} v_i x_i$

##### Constraints

1. Inventory capacity: $\sum_{i\in I} w_i x_i \leq C$.

2. Domains: $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.

##### Parameters

$I=\{\text{Sedan},\text{SUV},\text{Truck},\text{Convertible},\text{Minivan},\text{Coupe},\text{Hatchback},\text{Station Wagon},\text{Electric Car},\text{Hybrid Car},\text{Luxury Sedan},\text{Sports Car},\text{Crossover},\text{Diesel Truck},\text{Compact SUV},\text{Luxury SUV},\text{Cargo Van},\text{Pickup Truck},\text{Roadster},\text{Muscle Car},\text{Off-road Vehicle},\text{Camper Van},\text{Compact Car},\text{Motorcycle},\text{Electric SUV}\}$

$v_{\text{Sedan}}=1752,\; v_{\text{SUV}}=1856,\; v_{\text{Truck}}=8372,\; v_{\text{Convertible}}=6168,\; v_{\text{Minivan}}=9681,\; v_{\text{Coupe}}=8062,\; v_{\text{Hatchback}}=3895,\; v_{\text{Station Wagon}}=3254,\; v_{\text{Electric Car}}=1701,\; v_{\text{Hybrid Car}}=6799,\; v_{\text{Luxury Sedan}}=2724,\; v_{\text{Sports Car}}=6304,\; v_{\text{Crossover}}=3255,\; v_{\text{Diesel Truck}}=1923,\; v_{\text{Compact SUV}}=4103,\; v_{\text{Luxury SUV}}=4429,\; v_{\text{Cargo Van}}=2663,\; v_{\text{Pickup Truck}}=1691,\; v_{\text{Roadster}}=5632,\; v_{\text{Muscle Car}}=4793,\; v_{\text{Off-road Vehicle}}=1343,\; v_{\text{Camper Van}}=9124,\; v_{\text{Compact Car}}=3652,\; v_{\text{Motorcycle}}=8842,\; v_{\text{Electric SUV}}=9176$

$w_{\text{Sedan}}=15,\; w_{\text{SUV}}=87,\; w_{\text{Truck}}=36,\; w_{\text{Convertible}}=30,\; w_{\text{Minivan}}=33,\; w_{\text{Coupe}}=72,\; w_{\text{Hatchback}}=75,\; w_{\text{Station Wagon}}=71,\; w_{\text{Electric Car}}=51,\; w_{\text{Hybrid Car}}=21,\; w_{\text{Luxury Sedan}}=97,\; w_{\text{Sports Car}}=52,\; w_{\text{Crossover}}=25,\; w_{\text{Diesel Truck}}=15,\; w_{\text{Compact SUV}}=54,\; w_{\text{Luxury SUV}}=57,\; w_{\text{Cargo Van}}=18,\; w_{\text{Pickup Truck}}=69,\; w_{\text{Roadster}}=26,\; w_{\text{Muscle Car}}=38,\; w_{\text{Off-road Vehicle}}=31,\; w_{\text{Camper Van}}=74,\; w_{\text{Compact Car}}=82,\; w_{\text{Motorcycle}}=49,\; w_{\text{Electric SUV}}=64$

$C=1576$

###### Retrieved Information

{
  "capacity": 1576,
  "products": [
    {"ProductName":"Sedan","Value":1752,"Weight":15},
    {"ProductName":"SUV","Value":1856,"Weight":87},
    {"ProductName":"Truck","Value":8372,"Weight":36},
    {"ProductName":"Convertible","Value":6168,"Weight":30},
    {"ProductName":"Minivan","Value":9681,"Weight":33},
    {"ProductName":"Coupe","Value":8062,"Weight":72},
    {"ProductName":"Hatchback","Value":3895,"Weight":75},
    {"ProductName":"Station Wagon","Value":3254,"Weight":71},
    {"ProductName":"Electric Car","Value":1701,"Weight":51},
    {"ProductName":"Hybrid Car","Value":6799,"Weight":21},
    {"ProductName":"Luxury Sedan","Value":2724,"Weight":97},
    {"ProductName":"Sports Car","Value":6304,"Weight":52},
    {"ProductName":"Crossover","Value":3255,"Weight":25},
    {"ProductName":"Diesel Truck","Value":1923,"Weight":15},
    {"ProductName":"Compact SUV","Value":4103,"Weight":54},
    {"ProductName":"Luxury SUV","Value":4429,"Weight":57},
    {"ProductName":"Cargo Van","Value":2663,"Weight":18},
    {"ProductName":"Pickup Truck","Value":1691,"Weight":69},
    {"ProductName":"Roadster","Value":5632,"Weight":26},
    {"ProductName":"Muscle Car","Value":4793,"Weight":38},
    {"ProductName":"Off-road Vehicle","Value":1343,"Weight":31},
    {"ProductName":"Camper Van","Value":9124,"Weight":74},
    {"ProductName":"Compact Car","Value":3652,"Weight":82},
    {"ProductName":"Motorcycle","Value":8842,"Weight":49},
    {"ProductName":"Electric SUV","Value":9176,"Weight":64}
  ]
}