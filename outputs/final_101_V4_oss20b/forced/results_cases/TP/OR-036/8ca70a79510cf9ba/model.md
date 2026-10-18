##### Decision Variables

$x_{p}\in \mathbb{Z}_{\geq0}$: number of units of vehicle type $p$ to be ordered daily.

##### Objective Function

$\displaystyle \max \sum_{p} v_{p}\,x_{p}$

##### Constraints

1. Inventory capacity: $\displaystyle \sum_{p} w_{p}\,x_{p}\;\leq\;C$  
2. Non‑negativity and integrality: $\displaystyle x_{p}\;\geq\;0,\;\;x_{p}\in \mathbb{Z}$ for all $p$.

$C=1576$ is the total inventory capacity.  
The set of vehicle types $p$ is indexed by the order in which they appear in the data below.

###### Retrieved Information

```json
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
```