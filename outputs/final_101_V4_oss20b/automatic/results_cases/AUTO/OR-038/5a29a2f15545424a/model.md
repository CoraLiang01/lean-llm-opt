##### Objective Function: 

$ \quad \quad \max \sum_{i=1}^{10} p_i \cdot x_i $

##### Constraints

###### 1. Individual Vehicle Type Capacity Constraints: 

$ \quad \quad x_i \leq c_i , \quad \forall i=1,\dots ,10 $

###### 2. Total Inventory Capacity Constraint: 

$ \quad \quad \sum_{i=1}^{10} x_i \leq \sum_{i=1}^{10} c_i $

###### 3. Non-negativity and Integrality Constraints: 

$ \quad \quad x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\dots ,10 $

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
  "capacity": {
    "Sedans": 100,
    "SUVs": 80,
    "Electric Vehicles": 120,
    "Hybrid Vehicles": 90,
    "Trucks": 50,
    "Sports Cars": 30,
    "Compact Cars": 110,
    "Luxury Sedans": 40,
    "Vans": 60,
    "Pickup Trucks": 35
  }
}