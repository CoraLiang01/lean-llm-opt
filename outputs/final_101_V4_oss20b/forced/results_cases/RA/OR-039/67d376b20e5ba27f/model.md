##### Objective Function: 

$ \quad \quad \max \sum_{i \in \text{Products}} \sum_{j \in \text{Warehouses}} \text{Value}_i \cdot x_{ij} $

##### Constraints

###### 1. Capacity Constraints: 

$ \quad \quad \sum_{i \in \text{Products}} \text{Weight}_i \cdot x_{ij} \leq \text{Capacity}_j , \quad \forall j \in \text{Warehouses} $

###### 2. Non‑negativity and Integrality Constraints: 

$ \quad \quad x_{ij} \in \mathbb{Z}_{\geq 0} , \quad \forall i \in \text{Products}, \; \forall j \in \text{Warehouses} $

###### Retrieved Information

{
  "items": [
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
  "profit": {
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
  ],
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
  }
}