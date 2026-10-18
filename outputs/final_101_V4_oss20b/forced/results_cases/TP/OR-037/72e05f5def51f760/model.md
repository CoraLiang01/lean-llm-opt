##### Decision Variables

$x_i\geq0$: number of vehicles of type $i$ to be ordered per day (continuous).

##### Objective Function

$\max\sum_{i\in I}v_i\,x_i$

##### Constraints

1. Capacity: $\sum_{i\in I}w_i\,x_i\leq C$.

2. Non‑negativity: $x_i\geq0,\quad\forall i\in I$.

$C=765$ is the overall inventory capacity.  
$I=\{\text{Sedan},\text{SUV},\text{Truck},\text{Convertible},\text{Minivan},\text{Coupe},\text{Hatchback},\text{Station Wagon},\text{Electric Car},\text{Hybrid Car},\text{Luxury Sedan},\text{Sports Car},\text{Crossover},\text{Diesel Truck},\text{Compact SUV},\text{Luxury SUV},\text{Cargo Van},\text{Pickup Truck},\text{Roadster},\text{Muscle Car},\text{Off‑road Vehicle},\text{Camper Van},\text{Compact Car},\text{Motorcycle},\text{Electric SUV}\}$.

###### Retrieved Information

{
  "capacity": 765,
  "products": [
    {"name":"Sedan","value":2524,"weight":99},
    {"name":"SUV","value":4614,"weight":55},
    {"name":"Truck","value":8416,"weight":75},
    {"name":"Convertible","value":5917,"weight":94},
    {"name":"Minivan","value":9048,"weight":80},
    {"name":"Coupe","value":1140,"weight":82},
    {"name":"Hatchback","value":8962,"weight":71},
    {"name":"Station Wagon","value":1888,"weight":100},
    {"name":"Electric Car","value":8487,"weight":28},
    {"name":"Hybrid Car","value":4425,"weight":93},
    {"name":"Luxury Sedan","value":4717,"weight":84},
    {"name":"Sports Car","value":4210,"weight":83},
    {"name":"Crossover","value":1226,"weight":62},
    {"name":"Diesel Truck","value":7400,"weight":90},
    {"name":"Compact SUV","value":4639,"weight":99},
    {"name":"Luxury SUV","value":7712,"weight":96},
    {"name":"Cargo Van","value":3299,"weight":21},
    {"name":"Pickup Truck","value":9895,"weight":39},
    {"name":"Roadster","value":4496,"weight":99},
    {"name":"Muscle Car","value":4526,"weight":81},
    {"name":"Off‑road Vehicle","value":5688,"weight":6},
    {"name":"Camper Van","value":3007,"weight":58},
    {"name":"Compact Car","value":3623,"weight":37},
    {"name":"Motorcycle","value":8474,"weight":15},
    {"name":"Electric SUV","value":8372,"weight":37}
  ]
}