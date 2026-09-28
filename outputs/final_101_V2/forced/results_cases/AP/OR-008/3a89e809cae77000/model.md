##### Objective Function:

$\quad \min \sum_{i=1}^5 \sum_{j=1}^6 c_{ij} x_{ij}$

where $x_{ij}$ is the amount of produce shipped from Supplier $i$ to Customer $j$, and $c_{ij}$ is the transportation cost per unit from Supplier $i$ to Customer $j$.

##### Constraints

###### 1. Demand Satisfaction (for each customer):

$\sum_{i=1}^5 x_{ij} = d_j \quad \forall j \in \{1,2,3,4,5,6\}$

###### 2. Supply Capacity (for each supplier):

$\sum_{j=1}^6 x_{ij} \leq s_i \quad \forall i \in \{1,2,3,4,5\}$

###### 3. Non-negativity:

$x_{ij} \geq 0 \quad \forall i,j$

##### Retrieved Information

{
  "customers": [
    {"Customer1": 70},
    {"Customer2": 80},
    {"Customer3": 60},
    {"Customer4": 90},
    {"Customer5": 85},
    {"Customer6": 95}
  ],
  "suppliers": [
    {"Supplier1": 200},
    {"Supplier2": 250},
    {"Supplier3": 230},
    {"Supplier4": 220},
    {"Supplier5": 210}
  ],
  "transportation_costs": {
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

##### Parameter Definitions

- $x_{ij}$: Amount shipped from Supplier $i$ to Customer $j$
- $c_{ij}$: Transportation cost per unit from Supplier $i$ to Customer $j$
- $d_j$: Demand of Customer $j$
- $s_i$: Supply capacity of Supplier $i$

##### Explicit Data

- Customers and demands:
  - Customer1: $d_1 = 70$
  - Customer2: $d_2 = 80$
  - Customer3: $d_3 = 60$
  - Customer4: $d_4 = 90$
  - Customer5: $d_5 = 85$
  - Customer6: $d_6 = 95$

- Suppliers and capacities:
  - Supplier1: $s_1 = 200$
  - Supplier2: $s_2 = 250$
  - Supplier3: $s_3 = 230$
  - Supplier4: $s_4 = 220$
  - Supplier5: $s_5 = 210$

- Transportation cost matrix $[c_{ij}]$:

|            | Customer1 | Customer2 | Customer3 | Customer4 | Customer5 | Customer6 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Supplier1  |     2     |     3     |     1     |     2     |     3     |     2     |
| Supplier2  |     1     |     2     |     3     |     2     |     3     |     2     |
| Supplier3  |     3     |     1     |     2     |     3     |     2     |     3     |
| Supplier4  |     2     |     3     |     2     |     1     |     3     |     4     |
| Supplier5  |     3     |     2     |     3     |     3     |     2     |     3     |