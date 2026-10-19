##### Objective Function: 

$ \quad \quad \max \sum_{i=1}^{10} \sum_{j=1}^{10} v_{i} \, x_{ij} $

##### Constraints

###### 1. Capacity Constraints: 

$ \sum_{i=1}^{10} w_{i} \, x_{ij} \leq C_{j} \quad \forall j \in \{1,2,\dots ,10\} $

###### 2. Variable Constraints: 

$ x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,2,\dots ,10\}, \; \forall j \in \{1,2,\dots ,10\} $

###### Retrieved Information

{
  "value": {
    "Sedans": 1200,
    "SUVs": 1800,
    "Electric Vehicles": 2500,
    "Hybrid Vehicles": 2000,
    "Trucks": 1500,
    "Sports Cars": 3000,
    "Compact Cars": 1000,
    "Luxury Sedans": 3500,
    "Vans": 1600,
    "Pickup Trucks": 1700
  },
  "weight": {
    "Sedans": 20,
    "SUVs": 15,
    "Electric Vehicles": 25,
    "Hybrid Vehicles": 18,
    "Trucks": 10,
    "Sports Cars": 5,
    "Compact Cars": 22,
    "Luxury Sedans": 8,
    "Vans": 12,
    "Pickup Trucks": 7
  },
  "capacity": {
    "Warehouse 1": 100,
    "Warehouse 2": 80,
    "Warehouse 3": 120,
    "Warehouse 4": 90,
    "Warehouse 5": 50,
    "Warehouse 6": 30,
    "Warehouse 7": 110,
    "Warehouse 8": 40,
    "Warehouse 9": 60,
    "Warehouse 10": 35
  },
  "products": [
    "Sedans",
    "SUVs",
    "Electric Vehicles",
    "Hybrid Vehicles",
    "Trucks",
    "Sports Cars",
    "Compact Cars",
    "Luxury Sedans",
    "Vans",
    "Pickup Trucks"
  ],
  "warehouses": [
    "Warehouse 1",
    "Warehouse 2",
    "Warehouse 3",
    "Warehouse 4",
    "Warehouse 5",
    "Warehouse 6",
    "Warehouse 7",
    "Warehouse 8",
    "Warehouse 9",
    "Warehouse 10"
  ]
}