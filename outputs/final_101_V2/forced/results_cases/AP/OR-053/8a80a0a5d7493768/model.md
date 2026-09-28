##### Decision Variables

Let $x_{ij}$ denote the number of units of product $j$ to be placed on shelf $i$, where $i \in \{1,2,\ldots,10\}$ (shelves) and $j \in \{1,2,\ldots,20\}$ (products).

##### Parameters

- Shelf capacities:
  - $C_1 = 500$
  - $C_2 = 700$
  - $C_3 = 600$
  - $C_4 = 800$
  - $C_5 = 550$
  - $C_6 = 900$
  - $C_7 = 650$
  - $C_8 = 750$
  - $C_9 = 820$
  - $C_{10} = 570$

- Product values and weights:

| $j$ | Value $v_j$ | Weight $w_j$ |
|-----|------------|--------------|
| 1   | 50         | 10           |
| 2   | 70         | 20           |
| 3   | 30         | 5            |
| 4   | 60         | 15           |
| 5   | 80         | 25           |
| 6   | 90         | 30           |
| 7   | 40         | 12           |
| 8   | 100        | 35           |
| 9   | 55         | 10           |
| 10  | 75         | 20           |
| 11  | 65         | 18           |
| 12  | 95         | 28           |
| 13  | 45         | 8            |
| 14  | 85         | 22           |
| 15  | 70         | 25           |
| 16  | 110        | 40           |
| 17  | 50         | 14           |
| 18  | 60         | 16           |
| 19  | 120        | 50           |
| 20  | 100        | 30           |

##### Objective Function

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}$

##### Constraints

###### 1. Shelf Capacity Constraints

For each shelf $i$:

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

###### 2. Variable Constraints

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,\ldots,10\}, \; j \in \{1,2,\ldots,20\}$

##### Retrieved Information

{
  "shelves": {
    "1": 500,
    "2": 700,
    "3": 600,
    "4": 800,
    "5": 550,
    "6": 900,
    "7": 650,
    "8": 750,
    "9": 820,
    "10": 570
  },
  "products": {
    "1": {"Value": 50, "Weight": 10},
    "2": {"Value": 70, "Weight": 20},
    "3": {"Value": 30, "Weight": 5},
    "4": {"Value": 60, "Weight": 15},
    "5": {"Value": 80, "Weight": 25},
    "6": {"Value": 90, "Weight": 30},
    "7": {"Value": 40, "Weight": 12},
    "8": {"Value": 100, "Weight": 35},
    "9": {"Value": 55, "Weight": 10},
    "10": {"Value": 75, "Weight": 20},
    "11": {"Value": 65, "Weight": 18},
    "12": {"Value": 95, "Weight": 28},
    "13": {"Value": 45, "Weight": 8},
    "14": {"Value": 85, "Weight": 22},
    "15": {"Value": 70, "Weight": 25},
    "16": {"Value": 110, "Weight": 40},
    "17": {"Value": 50, "Weight": 14},
    "18": {"Value": 60, "Weight": 16},
    "19": {"Value": 120, "Weight": 50},
    "20": {"Value": 100, "Weight": 30}
  }
}