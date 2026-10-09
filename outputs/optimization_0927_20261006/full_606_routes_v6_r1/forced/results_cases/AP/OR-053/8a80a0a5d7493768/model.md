##### Objective Function:

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}$

where $x_{ij}$ is the integer number of units of product $j$ placed on shelf $i$, and $v_j$ is the value of product $j$.

##### Constraints:

###### 1. Shelf Capacity Constraints:

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

where $w_j$ is the weight of product $j$, and $C_i$ is the capacity of shelf $i$.

###### 2. Variable Constraints:

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

##### Indices

- Shelves $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- Products $j \in \{1,2,3,\ldots,20\}$

##### Parameters

- $C_i$: Shelf capacities

  $C_1 = 500$, $C_2 = 700$, $C_3 = 600$, $C_4 = 800$, $C_5 = 550$, $C_6 = 900$, $C_7 = 650$, $C_8 = 750$, $C_9 = 820$, $C_{10} = 570$

- $v_j$: Product values

  $v_1 = 50$, $v_2 = 70$, $v_3 = 30$, $v_4 = 60$, $v_5 = 80$, $v_6 = 90$, $v_7 = 40$, $v_8 = 100$, $v_9 = 55$, $v_{10} = 75$, $v_{11} = 65$, $v_{12} = 95$, $v_{13} = 45$, $v_{14} = 85$, $v_{15} = 70$, $v_{16} = 110$, $v_{17} = 50$, $v_{18} = 60$, $v_{19} = 120$, $v_{20} = 100$

- $w_j$: Product weights

  $w_1 = 10$, $w_2 = 20$, $w_3 = 5$, $w_4 = 15$, $w_5 = 25$, $w_6 = 30$, $w_7 = 12$, $w_8 = 35$, $w_9 = 10$, $w_{10} = 20$, $w_{11} = 18$, $w_{12} = 28$, $w_{13} = 8$, $w_{14} = 22$, $w_{15} = 25$, $w_{16} = 40$, $w_{17} = 14$, $w_{18} = 16$, $w_{19} = 50$, $w_{20} = 30$