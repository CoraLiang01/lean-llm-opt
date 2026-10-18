##### Sets and Indices

- Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of shelves, indexed by $i$.
- Let $J = \{1,2,\ldots,20\}$ be the set of products, indexed by $j$.

##### Parameters

- Shelf capacities:
  - $C_1 = 750$
  - $C_2 = 820$
  - $C_3 = 570$
  - $C_4 = 800$
  - $C_5 = 550$
  - $C_6 = 900$
  - $C_7 = 650$
  - $C_8 = 800$
  - $C_9 = 850$
  - $C_{10} = 900$

- Product values and weights:

| $j$ | $v_j$ (Value) | $w_j$ (Weight) |
|----|--------------|---------------|
| 1  | 55           | 10            |
| 2  | 75           | 20            |
| 3  | 65           | 5             |
| 4  | 60           | 15            |
| 5  | 80           | 25            |
| 6  | 90           | 35            |
| 7  | 40           | 45            |
| 8  | 100          | 55            |
| 9  | 55           | 65            |
| 10 | 75           | 20            |
| 11 | 110          | 18            |
| 12 | 50           | 28            |
| 13 | 60           | 8             |
| 14 | 120          | 28            |
| 15 | 70           | 25            |
| 16 | 110          | 40            |
| 17 | 50           | 55            |
| 18 | 60           | 70            |
| 19 | 120          | 85            |
| 20 | 100          | 100           |

##### Decision Variables

- $x_{ij} \geq 0$ : Number of units of product $j$ placed on shelf $i$ (integer, if only whole units are allowed; otherwise, continuous).

##### Objective Function

$\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}$

##### Constraints

1. **Shelf Capacity Constraints:**

$\sum_{j=1}^{20} w_j x_{ij} \leq C_i \quad \forall i \in I$

2. **Non-negativity:**

$x_{ij} \geq 0 \quad \forall i \in I, \forall j \in J$

##### Retrieved Information

{
  "shelves": {
    "1": 750,
    "2": 820,
    "3": 570,
    "4": 800,
    "5": 550,
    "6": 900,
    "7": 650,
    "8": 800,
    "9": 850,
    "10": 900
  },
  "products": {
    "1": {"Value": 55, "Weight": 10},
    "2": {"Value": 75, "Weight": 20},
    "3": {"Value": 65, "Weight": 5},
    "4": {"Value": 60, "Weight": 15},
    "5": {"Value": 80, "Weight": 25},
    "6": {"Value": 90, "Weight": 35},
    "7": {"Value": 40, "Weight": 45},
    "8": {"Value": 100, "Weight": 55},
    "9": {"Value": 55, "Weight": 65},
    "10": {"Value": 75, "Weight": 20},
    "11": {"Value": 110, "Weight": 18},
    "12": {"Value": 50, "Weight": 28},
    "13": {"Value": 60, "Weight": 8},
    "14": {"Value": 120, "Weight": 28},
    "15": {"Value": 70, "Weight": 25},
    "16": {"Value": 110, "Weight": 40},
    "17": {"Value": 50, "Weight": 55},
    "18": {"Value": 60, "Weight": 70},
    "19": {"Value": 120, "Weight": 85},
    "20": {"Value": 100, "Weight": 100}
  }
}