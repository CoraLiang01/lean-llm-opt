##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: number of vehicles of type $i$ to be ordered per day, for each $i \in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I} b_i x_i$

##### Constraints

1. Vehicle‑type capacity limits: $\displaystyle x_i \leq c_i,\quad \forall i\in I$  
2. Total inventory capacity: $\displaystyle \sum_{i\in I} x_i \leq M$  
3. Domains: $x_i \in \mathbb{Z}_{\geq 0}$ for all $i\in I$.

$I=\{1,2,3,4,5,6,7,8,9,10\}$,  
$M=\sum_{i\in I} c_i=715$.

##### Retrieved Information

{
  "VehicleIDs": [1,2,3,4,5,6,7,8,9,10],
  "VehicleTypes": ["Sedans","SUVs","Electric Vehicles","Hybrid Vehicles","Trucks","Sports Cars","Compact Cars","Luxury Sedans","Vans","Pickup Trucks"],
  "Capacity": [100,80,120,90,50,30,110,40,60,35],
  "Benefit": [1200,1800,2500,2000,1500,3000,1000,3500,1600,1700],
  "TotalCapacity": 715
}