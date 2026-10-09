##### Objective Function:

$\quad \max \sum_{i=1}^8 \sum_{j=1}^{10} v_j \, x_{ij}$

where $x_{ij}$ is the number of units of product $j$ placed in section $i$, and $v_j$ is the value (price) of product $j$.

##### Constraints:

###### 1. Section Capacity Constraints:

$\sum_{j=1}^{10} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,3,4,5,6,7,8\}$

where $w_j$ is the shelf space requirement (weight) of product $j$, and $C_i$ is the capacity of section $i$.

###### 2. Variable Domain Constraints:

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,3,4,5,6,7,8\}, \; j \in \{1,2,3,4,5,6,7,8,9,10\}$

##### Retrieved Information

{
  "sections": [
    {"SectionID": "1", "Capacity": 100},
    {"SectionID": "2", "Capacity": 150},
    {"SectionID": "3", "Capacity": 120},
    {"SectionID": "4", "Capacity": 130},
    {"SectionID": "5", "Capacity": 90},
    {"SectionID": "6", "Capacity": 110},
    {"SectionID": "7", "Capacity": 160},
    {"SectionID": "8", "Capacity": 140}
  ],
  "products": [
    {"ProductName": "1", "Value": 10, "Weight": 2},
    {"ProductName": "2", "Value": 15, "Weight": 3},
    {"ProductName": "3", "Value": 8, "Weight": 1},
    {"ProductName": "4", "Value": 12, "Weight": 2},
    {"ProductName": "5", "Value": 20, "Weight": 4},
    {"ProductName": "6", "Value": 25, "Weight": 5},
    {"ProductName": "7", "Value": 5, "Weight": 1},
    {"ProductName": "8", "Value": 30, "Weight": 6},
    {"ProductName": "9", "Value": 18, "Weight": 3},
    {"ProductName": "10", "Value": 22, "Weight": 4}
  ]
}

- Section capacities: $C_1=100$, $C_2=150$, $C_3=120$, $C_4=130$, $C_5=90$, $C_6=110$, $C_7=160$, $C_8=140$
- Product values: $v_1=10$, $v_2=15$, $v_3=8$, $v_4=12$, $v_5=20$, $v_6=25$, $v_7=5$, $v_8=30$, $v_9=18$, $v_{10}=22$
- Product weights: $w_1=2$, $w_2=3$, $w_3=1$, $w_4=2$, $w_5=4$, $w_6=5$, $w_7=1$, $w_8=6$, $w_9=3$, $w_{10}=4$

##### Decision Variables

$x_{ij}$: integer, number of units of product $j$ to be placed in section $i$.

##### Full Model

$\max \sum_{i=1}^8 \sum_{j=1}^{10} v_j \, x_{ij}$

subject to

$\sum_{j=1}^{10} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,3,4,5,6,7,8\}$

$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,3,4,5,6,7,8\}, \; j \in \{1,2,3,4,5,6,7,8,9,10\}$