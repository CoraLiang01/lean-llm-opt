##### Decision Variables

$x_{ij} \geq 0$: number of units of product $j \in J$ placed on shelf $i \in I$ (continuous or integer, depending on context).

##### Parameters

- Shelves $I = \{1,2,3,4,5,6,7,8,9,10\}$
- Products $J = \{1,2,3,\ldots,20\}$

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
  - $v_1 = 55$, $w_1 = 10$
  - $v_2 = 75$, $w_2 = 20$
  - $v_3 = 65$, $w_3 = 5$
  - $v_4 = 60$, $w_4 = 15$
  - $v_5 = 80$, $w_5 = 25$
  - $v_6 = 90$, $w_6 = 35$
  - $v_7 = 40$, $w_7 = 45$
  - $v_8 = 100$, $w_8 = 55$
  - $v_9 = 55$, $w_9 = 65$
  - $v_{10} = 75$, $w_{10} = 20$
  - $v_{11} = 110$, $w_{11} = 18$
  - $v_{12} = 50$, $w_{12} = 28$
  - $v_{13} = 60$, $w_{13} = 8$
  - $v_{14} = 120$, $w_{14} = 28$
  - $v_{15} = 70$, $w_{15} = 25$
  - $v_{16} = 110$, $w_{16} = 40$
  - $v_{17} = 50$, $w_{17} = 55$
  - $v_{18} = 60$, $w_{18} = 70$
  - $v_{19} = 120$, $w_{19} = 85$
  - $v_{20} = 100$, $w_{20} = 100$

##### Objective Function

$\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}$

##### Constraints

1. Shelf capacity constraints:
   $$
   \sum_{j \in J} w_j x_{ij} \leq C_i, \quad \forall i \in I
   $$

2. Nonnegativity:
   $$
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   $$

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