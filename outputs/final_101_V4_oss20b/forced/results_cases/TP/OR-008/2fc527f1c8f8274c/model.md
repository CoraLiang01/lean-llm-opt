##### Decision Variables

$x_{ij}\geq0$: quantity shipped from supplier $i\in I$ to customer $j\in J$ (continuous).

##### Objective Function

$\min\sum_{i\in I}\sum_{j\in J}c_{ij}x_{ij}$

##### Constraints

1. Demand: $\sum_{i\in I}x_{ij}\geq d_j,\quad\forall j\in J$.
2. Supply capacity: $\sum_{j\in J}x_{ij}\leq s_i,\quad\forall i\in I$.
3. Non‑negativity: $x_{ij}\geq0$ continuous.

$I=\{Supplier1,Supplier2,Supplier3,Supplier4,Supplier5\}$  
$J=\{Customer1,Customer2,Customer3,Customer4,Customer5,Customer6\}$  
The cost matrix $c_{ij}$, demand vector $d_j$, and supply capacities $s_i$ are given below.

###### Retrieved Information

{
  "suppliers": [
    "Supplier1",
    "Supplier2",
    "Supplier3",
    "Supplier4",
    "Supplier5"
  ],
  "customers": [
    "Customer1",
    "Customer2",
    "Customer3",
    "Customer4",
    "Customer5",
    "Customer6"
  ],
  "demand": {
    "Customer1": 70,
    "Customer2": 80,
    "Customer3": 60,
    "Customer4": 90,
    "Customer5": 85,
    "Customer6": 95
  },
  "supply_capacity": {
    "Supplier1": 200,
    "Supplier2": 250,
    "Supplier3": 230,
    "Supplier4": 220,
    "Supplier5": 210
  },
  "cost": {
    "Supplier1": {
      "Customer1": 2,
      "Customer2": 3,
      "Customer3": 1,
      "Customer4": 2,
      "Customer5": 3,
      "Customer6": 2
    },
    "Supplier2": {
      "Customer1": 1,
      "Customer2": 2,
      "Customer3": 3,
      "Customer4": 2,
      "Customer5": 3,
      "Customer6": 2
    },
    "Supplier3": {
      "Customer1": 3,
      "Customer2": 1,
      "Customer3": 2,
      "Customer4": 3,
      "Customer5": 2,
      "Customer6": 3
    },
    "Supplier4": {
      "Customer1": 2,
      "Customer2": 3,
      "Customer3": 2,
      "Customer4": 1,
      "Customer5": 3,
      "Customer6": 4
    },
    "Supplier5": {
      "Customer1": 3,
      "Customer2": 2,
      "Customer3": 3,
      "Customer4": 3,
      "Customer5": 2,
      "Customer6": 3
    }
  }
}