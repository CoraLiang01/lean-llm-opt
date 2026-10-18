##### Objective Function: 

$\quad \quad \max \sum_i \sum_j p_j \cdot x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$\quad \quad \sum_j w_j \, x_{ij} \leq c_i, \quad \forall i$

###### 2. Non-negativity and Integrality Constraints: 

$\quad \quad x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i,j$

I indexes the 15 storage areas; J indexes the 10 air‑conditioner types. $p_j$ is the Value of product $j$, $w_j$ is the Weight of product $j$, and $c_i$ is the Capacity of storage area $i$. All decision variables are nonnegative integers representing whole units.

###### Retrieved Information

{
  "items": [
    "Window Unit",
    "Portable Unit",
    "Split System",
    "Ductless System",
    "Central AC",
    "Hybrid AC",
    "Geothermal AC",
    "Smart AC",
    "Evaporative Cooler",
    "Package Unit"
  ],
  "profit": {
    "Window Unit": 4811,
    "Portable Unit": 1130,
    "Split System": 1611,
    "Ductless System": 3368,
    "Central AC": 2135,
    "Hybrid AC": 1046,
    "Geothermal AC": 4030,
    "Smart AC": 3761,
    "Evaporative Cooler": 3523,
    "Package Unit": 1701
  },
  "shelves": [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15"
  ],
  "capacity": {
    "1": 1083,
    "2": 1840,
    "3": 770,
    "4": 1299,
    "5": 1259,
    "6": 543,
    "7": 1831,
    "8": 855,
    "9": 619,
    "10": 637,
    "11": 935,
    "12": 626,
    "13": 1457,
    "14": 1198,
    "15": 837
  },
  "weight": {
    "Window Unit": 114,
    "Portable Unit": 200,
    "Split System": 106,
    "Ductless System": 256,
    "Central AC": 268,
    "Hybrid AC": 185,
    "Geothermal AC": 299,
    "Smart AC": 131,
    "Evaporative Cooler": 139,
    "Package Unit": 105
  }
}