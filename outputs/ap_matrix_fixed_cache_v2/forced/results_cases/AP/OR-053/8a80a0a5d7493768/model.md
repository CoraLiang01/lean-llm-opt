##### Decision Variables

Let $x_{ij}$ denote the number of units of product $j$ placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

##### Parameters

- Shelves $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- Products $j \in \{1,2,\ldots,20\}$

Shelf capacities:
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

Product values and weights:
- Product 1: $v_1 = 50$, $w_1 = 10$
- Product 2: $v_2 = 70$, $w_2 = 20$
- Product 3: $v_3 = 30$, $w_3 = 5$
- Product 4: $v_4 = 60$, $w_4 = 15$
- Product 5: $v_5 = 80$, $w_5 = 25$
- Product 6: $v_6 = 90$, $w_6 = 30$
- Product 7: $v_7 = 40$, $w_7 = 12$
- Product 8: $v_8 = 100$, $w_8 = 35$
- Product 9: $v_9 = 55$, $w_9 = 10$
- Product 10: $v_{10} = 75$, $w_{10} = 20$
- Product 11: $v_{11} = 65$, $w_{11} = 18$
- Product 12: $v_{12} = 95$, $w_{12} = 28$
- Product 13: $v_{13} = 45$, $w_{13} = 8$
- Product 14: $v_{14} = 85$, $w_{14} = 22$
- Product 15: $v_{15} = 70$, $w_{15} = 25$
- Product 16: $v_{16} = 110$, $w_{16} = 40$
- Product 17: $v_{17} = 50$, $w_{17} = 14$
- Product 18: $v_{18} = 60$, $w_{18} = 16$
- Product 19: $v_{19} = 120$, $w_{19} = 50$
- Product 20: $v_{20} = 100$, $w_{20} = 30$

##### Objective Function

$\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}$

##### Constraints

For each shelf $i = 1, \ldots, 10$:
$$
\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i
$$

For all $i = 1, \ldots, 10$ and $j = 1, \ldots, 20$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

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