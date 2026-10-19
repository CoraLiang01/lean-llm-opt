##### Decision Variables

$x_i\in \mathbb{Z}_{\ge 0}$: number of vehicles of type $i$ ordered per day, for $i\in I$.

##### Objective Function

$\max \sum_{i\in I}v_i\,x_i$

##### Constraints

1. Per‑type capacity: $\displaystyle x_i\leq c_i,\quad\forall i\in I$  
2. Total inventory capacity: $\displaystyle \sum_{i\in I}x_i\leq \sum_{i\in I}c_i$  
3. Integer non‑negativity: $x_i\in \mathbb{Z}_{\ge 0},\quad\forall i\in I$

$I=\{1,2,3,4,5,6,7,8,9,10\}$, where each index $i$ corresponds to the vehicle type listed in the tables below.

###### Retrieved Information

```json
{
  "capacity": [
    {"VehicleID":"1","VehicleType":"Sedans","Capacity":"100"},
    {"VehicleID":"2","VehicleType":"SUVs","Capacity":"80"},
    {"VehicleID":"3","VehicleType":"Electric Vehicles","Capacity":"120"},
    {"VehicleID":"4","VehicleType":"Hybrid Vehicles","Capacity":"90"},
    {"VehicleID":"5","VehicleType":"Trucks","Capacity":"50"},
    {"VehicleID":"6","VehicleType":"Sports Cars","Capacity":"30"},
    {"VehicleID":"7","VehicleType":"Compact Cars","Capacity":"110"},
    {"VehicleID":"8","VehicleType":"Luxury Sedans","Capacity":"40"},
    {"VehicleID":"9","VehicleType":"Vans","Capacity":"60"},
    {"VehicleID":"10","VehicleType":"Pickup Trucks","Capacity":"35"}
  ],
  "products": [
    {"ProductName":"Sedans","Value":"1200","Weight":"20"},
    {"ProductName":"SUVs","Value":"1800","Weight":"15"},
    {"ProductName":"Electric Vehicles","Value":"2500","Weight":"25"},
    {"ProductName":"Hybrid Vehicles","Value":"2000","Weight":"18"},
    {"ProductName":"Trucks","Value":"1500","Weight":"10"},
    {"ProductName":"Sports Cars","Value":"3000","Weight":"5"},
    {"ProductName":"Compact Cars","Value":"1000","Weight":"22"},
    {"ProductName":"Luxury Sedans","Value":"3500","Weight":"8"},
    {"ProductName":"Vans","Value":"1600","Weight":"12"},
    {"ProductName":"Pickup Trucks","Value":"1700","Weight":"7"}
  ]
}
```