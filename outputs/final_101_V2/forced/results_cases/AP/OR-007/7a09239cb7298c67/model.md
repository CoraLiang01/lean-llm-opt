##### Objective Function:

$\quad \min \sum_{i=1}^5 \sum_{j=1}^5 c_{ij} x_{ij}$

where $c_{ij}$ is the cost of transporting one unit from warehouse $S_i$ to store $D_j$, and $x_{ij}$ is the number of units shipped from $S_i$ to $D_j$.

##### Constraints

###### 1. Demand Satisfaction (for each store):

$\sum_{i=1}^5 x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5\}$

###### 2. Supply Capacity (for each warehouse):

$\sum_{j=1}^5 x_{ij} \leq s_i \quad \forall i \in \{1,2,3,4,5\}$

###### 3. Non-negativity:

$x_{ij} \geq 0 \quad \forall i,j$

##### Retrieved Information

{
  "warehouses": ["S1", "S2", "S3", "S4", "S5"],
  "stores": ["D1", "D2", "D3", "D4", "D5"],
  "demand": {
    "D1": 428,
    "D2": 217,
    "D3": 214,
    "D4": 380,
    "D5": 254
  },
  "supply_capacity": {
    "S1": 428,
    "S2": 217,
    "S3": 214,
    "S4": 380,
    "S5": 254
  },
  "transportation_costs": {
    "S1": {
      "D1": 269.3910588020795,
      "D2": 1.4537335390933939,
      "D3": 99.60345345756605,
      "D4": 26.64078166309837,
      "D5": 9.537688956880922
    },
    "S2": {
      "D1": 9.291846876785183,
      "D2": 10.874778437070223,
      "D3": 144.52609291614627,
      "D4": 11.420133077898234,
      "D5": 153.1756819927813
    },
    "S3": {
      "D1": 9.674584301671008,
      "D2": 2.6191650959687944,
      "D3": 100.8242249168735,
      "D4": 3.2121910887916876,
      "D5": 133.8493396124168
    },
    "S4": {
      "D1": 270.57498480010247,
      "D2": 32.50253586,
      "D3": 4.6842098096469815,
      "D4": 1.5682269686546804,
      "D5": 9.58927599
    },
    "S5": {
      "D1": 226.0331910675782,
      "D2": 8.669161980826471,
      "D3": 65.47681316968448,
      "D4": 9.068765258459958,
      "D5": 202.65015316425533
    }
  }
}

##### Variable Definitions

$x_{ij}$: Number of units shipped from warehouse $S_i$ to store $D_j$, for $i \in \{1,2,3,4,5\}$ and $j \in \{1,2,3,4,5\}$.

$c_{ij}$: Transportation cost per unit from warehouse $S_i$ to store $D_j$, as given above.

$d_j$: Demand at store $D_j$, as given above.

$s_i$: Supply capacity at warehouse $S_i$, as given above.