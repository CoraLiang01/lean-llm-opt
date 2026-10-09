##### Objective Function:

$\quad \min \sum_{i \in S} \sum_{j \in D} c_{ij} \, x_{ij}$

where:
- $S = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses)
- $D = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores)
- $c_{ij}$ is the cost per unit from warehouse $i$ to store $j$
- $x_{ij}$ is the quantity shipped from warehouse $i$ to store $j$

##### Constraints

###### 1. Demand Satisfaction (each store's demand must be met):

$\sum_{i \in S} x_{ij} = d_j \quad \forall j \in D$

where $d_j$ is the demand for store $j$.

###### 2. Supply Capacity (each warehouse's supply cannot be exceeded):

$\sum_{j \in D} x_{ij} \leq s_i \quad \forall i \in S$

where $s_i$ is the supply capacity of warehouse $i$.

###### 3. Non-negativity:

$x_{ij} \geq 0 \quad \forall i \in S, \forall j \in D$

##### Retrieved Information

{
  "stores": [
    {"id": "D1", "demand": 428},
    {"id": "D2", "demand": 217},
    {"id": "D3", "demand": 214},
    {"id": "D4", "demand": 380},
    {"id": "D5", "demand": 254}
  ],
  "warehouses": [
    {"id": "S1", "supply_capacity": 428},
    {"id": "S2", "supply_capacity": 217},
    {"id": "S3", "supply_capacity": 214},
    {"id": "S4", "supply_capacity": 380},
    {"id": "S5", "supply_capacity": 254}
  ],
  "transportation_costs": {
    "S1": {"D1": 269.3910588020795, "D2": 1.4537335390933939, "D3": 99.60345345756605, "D4": 26.64078166309837, "D5": 9.537688956880922},
    "S2": {"D1": 9.291846876785183, "D2": 10.874778437070223, "D3": 144.52609291614627, "D4": 11.420133077898234, "D5": 153.1756819927813},
    "S3": {"D1": 9.674584301671008, "D2": 2.6191650959687944, "D3": 100.8242249168735, "D4": 3.2121910887916876, "D5": 133.8493396124168},
    "S4": {"D1": 270.57498480010247, "D2": 32.50253586, "D3": 4.6842098096469815, "D4": 1.5682269686546804, "D5": 9.58927599},
    "S5": {"D1": 226.0331910675782, "D2": 8.669161980826471, "D3": 65.47681316968448, "D4": 9.068765258459958, "D5": 202.65015316425533}
  }
}

##### Full Model with Parameters

Let $x_{ij}$ be the quantity shipped from warehouse $i$ to store $j$.

Minimize:
$$
269.3910588020795\,x_{\text{S1},\text{D1}} + 1.4537335390933939\,x_{\text{S1},\text{D2}} + 99.60345345756605\,x_{\text{S1},\text{D3}} + 26.64078166309837\,x_{\text{S1},\text{D4}} + 9.537688956880922\,x_{\text{S1},\text{D5}} \\
+ 9.291846876785183\,x_{\text{S2},\text{D1}} + 10.874778437070223\,x_{\text{S2},\text{D2}} + 144.52609291614627\,x_{\text{S2},\text{D3}} + 11.420133077898234\,x_{\text{S2},\text{D4}} + 153.1756819927813\,x_{\text{S2},\text{D5}} \\
+ 9.674584301671008\,x_{\text{S3},\text{D1}} + 2.6191650959687944\,x_{\text{S3},\text{D2}} + 100.8242249168735\,x_{\text{S3},\text{D3}} + 3.2121910887916876\,x_{\text{S3},\text{D4}} + 133.8493396124168\,x_{\text{S3},\text{D5}} \\
+ 270.57498480010247\,x_{\text{S4},\text{D1}} + 32.50253586\,x_{\text{S4},\text{D2}} + 4.6842098096469815\,x_{\text{S4},\text{D3}} + 1.5682269686546804\,x_{\text{S4},\text{D4}} + 9.58927599\,x_{\text{S4},\text{D5}} \\
+ 226.0331910675782\,x_{\text{S5},\text{D1}} + 8.669161980826471\,x_{\text{S5},\text{D2}} + 65.47681316968448\,x_{\text{S5},\text{D3}} + 9.068765258459958\,x_{\text{S5},\text{D4}} + 202.65015316425533\,x_{\text{S5},\text{D5}}
$$

Subject to:

$\quad x_{\text{S1},j} + x_{\text{S2},j} + x_{\text{S3},j} + x_{\text{S4},j} + x_{\text{S5},j} = d_j \quad \forall j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

- $x_{\text{S1},\text{D1}} + x_{\text{S2},\text{D1}} + x_{\text{S3},\text{D1}} + x_{\text{S4},\text{D1}} + x_{\text{S5},\text{D1}} = 428$
- $x_{\text{S1},\text{D2}} + x_{\text{S2},\text{D2}} + x_{\text{S3},\text{D2}} + x_{\text{S4},\text{D2}} + x_{\text{S5},\text{D2}} = 217$
- $x_{\text{S1},\text{D3}} + x_{\text{S2},\text{D3}} + x_{\text{S3},\text{D3}} + x_{\text{S4},\text{D3}} + x_{\text{S5},\text{D3}} = 214$
- $x_{\text{S1},\text{D4}} + x_{\text{S2},\text{D4}} + x_{\text{S3},\text{D4}} + x_{\text{S4},\text{D4}} + x_{\text{S5},\text{D4}} = 380$
- $x_{\text{S1},\text{D5}} + x_{\text{S2},\text{D5}} + x_{\text{S3},\text{D5}} + x_{\text{S4},\text{D5}} + x_{\text{S5},\text{D5}} = 254$

$\quad x_{i,\text{D1}} + x_{i,\text{D2}} + x_{i,\text{D3}} + x_{i,\text{D4}} + x_{i,\text{D5}} \leq s_i \quad \forall i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$

- $x_{\text{S1},\text{D1}} + x_{\text{S1},\text{D2}} + x_{\text{S1},\text{D3}} + x_{\text{S1},\text{D4}} + x_{\text{S1},\text{D5}} \leq 428$
- $x_{\text{S2},\text{D1}} + x_{\text{S2},\text{D2}} + x_{\text{S2},\text{D3}} + x_{\text{S2},\text{D4}} + x_{\text{S2},\text{D5}} \leq 217$
- $x_{\text{S3},\text{D1}} + x_{\text{S3},\text{D2}} + x_{\text{S3},\text{D3}} + x_{\text{S3},\text{D4}} + x_{\text{S3},\text{D5}} \leq 214$
- $x_{\text{S4},\text{D1}} + x_{\text{S4},\text{D2}} + x_{\text{S4},\text{D3}} + x_{\text{S4},\text{D4}} + x_{\text{S4},\text{D5}} \leq 380$
- $x_{\text{S5},\text{D1}} + x_{\text{S5},\text{D2}} + x_{\text{S5},\text{D3}} + x_{\text{S5},\text{D4}} + x_{\text{S5},\text{D5}} \leq 254$

$\quad x_{ij} \geq 0 \quad \forall i \in S, \forall j \in D$