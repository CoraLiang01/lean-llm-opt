##### Decision Variables

$x_i \ge 0$: number of vehicles of type $i$ to order per day.

##### Objective Function

$\max \sum_{i\in I} p_i x_i$

##### Constraints

1. Capacity constraint: $\sum_{i\in I} w_i x_i \le C$  
2. Non‑negativity: $x_i \ge 0,\quad \forall i\in I$

##### Parameters

$I=\{\text{Sedan},\text{SUV},\text{Truck},\text{Convertible},\text{Minivan},\text{Coupe},\text{Hatchback},\text{Station Wagon},\text{Electric Car},\text{Hybrid Car},\text{Luxury Sedan},\text{Sports Car},\text{Crossover},\text{Diesel Truck},\text{Compact SUV},\text{Luxury SUV},\text{Cargo Van},\text{Pickup Truck},\text{Roadster},\text{Muscle Car},\text{Off‑road Vehicle},\text{Camper Van},\text{Compact Car},\text{Motorcycle},\text{Electric SUV}\}$  

$p=\begin{bmatrix}
2524\\
4614\\
8416\\
5917\\
9048\\
1140\\
8962\\
1888\\
8487\\
4425\\
4717\\
4210\\
1226\\
7400\\
4639\\
7712\\
3299\\
9895\\
4496\\
4526\\
5688\\
3007\\
3623\\
8474\\
8372
\end{bmatrix}$  

$w=\begin{bmatrix}
99\\
55\\
75\\
94\\
80\\
82\\
71\\
100\\
28\\
93\\
84\\
83\\
62\\
90\\
99\\
96\\
21\\
39\\
99\\
81\\
6\\
58\\
37\\
15\\
37
\end{bmatrix}$  

$C=765$

###### Retrieved Information

{
  "capacity": 765,
  "products": [
    {"ProductName":"Sedan","Value":2524,"Weight":99},
    {"ProductName":"SUV","Value":4614,"Weight":55},
    {"ProductName":"Truck","Value":8416,"Weight":75},
    {"ProductName":"Convertible","Value":5917,"Weight":94},
    {"ProductName":"Minivan","Value":9048,"Weight":80},
    {"ProductName":"Coupe","Value":1140,"Weight":82},
    {"ProductName":"Hatchback","Value":8962,"Weight":71},
    {"ProductName":"Station Wagon","Value":1888,"Weight":100},
    {"ProductName":"Electric Car","Value":8487,"Weight":28},
    {"ProductName":"Hybrid Car","Value":4425,"Weight":93},
    {"ProductName":"Luxury Sedan","Value":4717,"Weight":84},
    {"ProductName":"Sports Car","Value":4210,"Weight":83},
    {"ProductName":"Crossover","Value":1226,"Weight":62},
    {"ProductName":"Diesel Truck","Value":7400,"Weight":90},
    {"ProductName":"Compact SUV","Value":4639,"Weight":99},
    {"ProductName":"Luxury SUV","Value":7712,"Weight":96},
    {"ProductName":"Cargo Van","Value":3299,"Weight":21},
    {"ProductName":"Pickup Truck","Value":9895,"Weight":39},
    {"ProductName":"Roadster","Value":4496,"Weight":99},
    {"ProductName":"Muscle Car","Value":4526,"Weight":81},
    {"ProductName":"Off‑road Vehicle","Value":5688,"Weight":6},
    {"ProductName":"Camper Van","Value":3007,"Weight":58},
    {"ProductName":"Compact Car","Value":3623,"Weight":37},
    {"ProductName":"Motorcycle","Value":8474,"Weight":15},
    {"ProductName":"Electric SUV","Value":8372,"Weight":37}
  ]
}