##### Objective Function: 

$ \quad \quad \max \sum_{i=1}^{10} v_i x_i $

##### Constraints

###### 1. Individual Capacity Constraints: 

$ x_i \leq c_i \quad \forall i \in \{1,\dots ,10\} $

###### 2. Total Capacity Constraint: 

$ \sum_{i=1}^{10} x_i \leq \sum_{i=1}^{10} c_i $

###### 3. Integer Constraints: 

$ x_i \in \mathbb{Z}_{\ge 0} \quad \forall i \in \{1,\dots ,10\} $

##### Retrieved Information

{
  "capacity": {
    "1": 100,
    "2": 80,
    "3": 120,
    "4": 90,
    "5": 50,
    "6": 30,
    "7": 110,
    "8": 40,
    "9": 60,
    "10": 35
  },
  "benefit": {
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
  "vehicle_types": [
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
  "vehicle_ids": [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10"
  ]
}