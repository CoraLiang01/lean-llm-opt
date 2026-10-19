##### Objective Function: 

$$\quad \max \sum_i \sum_j p_i \cdot x_{ij}$$

##### Constraints

###### 1. Capacity Constraints: 

$$\quad \sum_i a_i x_{ij} \leq c_j, \quad \forall j$$

###### 2. Non-negativity and Integrality Constraints: 

$$\quad  x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i,j $$

I indexes the five Sony products; J indexes PlatformID. p_i is Value, a_i is SpaceRequirement, and c_j is SonyLimit. Every SonyLimit is below the corresponding total Capacity in this reference. Variables are nonnegative integers representing whole product units.

###### Retrieved Information

{
  "items": [
    "Sony Alpha Refrigerator",
    "Sony Bravia XR",
    "Sony HT-A7000 Soundbar",
    "Sony PlayStation 5",
    "Sony PlayStation Soundbar"
  ],
  "profit": {
    "Sony Alpha Refrigerator": 1818,
    "Sony Bravia XR": 1609,
    "Sony HT-A7000 Soundbar": 509,
    "Sony PlayStation 5": 1808,
    "Sony PlayStation Soundbar": 528
  },
  "shelves": [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6"
  ],
  "sony_limit": {
    "1": 1200,
    "2": 1374,
    "3": 800,
    "4": 2042,
    "5": 1000,
    "6": 1800
  },
  "space": {
    "Sony Alpha Refrigerator": 400,
    "Sony Bravia XR": 200,
    "Sony HT-A7000 Soundbar": 40,
    "Sony PlayStation 5": 60,
    "Sony PlayStation Soundbar": 50
  }
}