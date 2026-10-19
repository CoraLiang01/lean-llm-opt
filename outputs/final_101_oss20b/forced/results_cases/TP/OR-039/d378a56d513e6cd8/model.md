##### Decision Variables

$x_i \in \mathbb{Z}_{\ge 0}$: number of vehicles of type $i$ to be ordered per day.

##### Objective Function

$\max \sum_{i} v_i\,x_i$

##### Constraints

1. Capacity: $\displaystyle \sum_{i} w_i\,x_i \;\le\; 715$

2. Non‑negativity and integrality: $x_i \in \mathbb{Z}_{\ge 0}$

where $i$ indexes the 10 vehicle types.

###### Retrieved Information

{
  "warehouses": [
    {"Warehouse ID":"Warehouse 1","Capacity":"100"},
    {"Warehouse ID":"Warehouse 2","Capacity":"80"},
    {"Warehouse ID":"Warehouse 3","Capacity":"120"},
    {"Warehouse ID":"Warehouse 4","Capacity":"90"},
    {"Warehouse ID":"Warehouse 5","Capacity":"50"},
    {"Warehouse ID":"Warehouse 6","Capacity":"30"},
    {"Warehouse ID":"Warehouse 7","Capacity":"110"},
    {"Warehouse ID":"Warehouse 8","Capacity":"40"},
    {"Warehouse ID":"Warehouse 9","Capacity":"60"},
    {"Warehouse ID":"Warehouse 10","Capacity":"35"}
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