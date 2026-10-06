##### Objective Function:

$\quad \min \sum_{i=1}^{12} \sum_{j=1}^{12} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j=1}^{12} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,12\}$

$\sum_{i=1}^{12} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,12\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\}, \quad \forall i,j$

##### Retrieved Information

{
  "cost": [
    ["machine_1_task_1", "machine_1_task_2", "machine_1_task_3", "machine_1_task_4", "machine_1_task_5", "machine_1_task_6", "machine_1_task_7", "machine_1_task_8", "machine_1_task_9", "machine_1_task_10", "machine_1_task_11", "machine_1_task_12"],
    ["machine_2_task_1", "machine_2_task_2", "machine_2_task_3", "machine_2_task_4", "machine_2_task_5", "machine_2_task_6", "machine_2_task_7", "machine_2_task_8", "machine_2_task_9", "machine_2_task_10", "machine_2_task_11", "machine_2_task_12"],
    ["machine_3_task_1", "machine_3_task_2", "machine_3_task_3", "machine_3_task_4", "machine_3_task_5", "machine_3_task_6", "machine_3_task_7", "machine_3_task_8", "machine_3_task_9", "machine_3_task_10", "machine_3_task_11", "machine_3_task_12"],
    ["machine_4_task_1", "machine_4_task_2", "machine_4_task_3", "machine_4_task_4", "machine_4_task_5", "machine_4_task_6", "machine_4_task_7", "machine_4_task_8", "machine_4_task_9", "machine_4_task_10", "machine_4_task_11", "machine_4_task_12"],
    ["machine_5_task_1", "machine_5_task_2", "machine_5_task_3", "machine_5_task_4", "machine_5_task_5", "machine_5_task_6", "machine_5_task_7", "machine_5_task_8", "machine_5_task_9", "machine_5_task_10", "machine_5_task_11", "machine_5_task_12"],
    ["machine_6_task_1", "machine_6_task_2", "machine_6_task_3", "machine_6_task_4", "machine_6_task_5", "machine_6_task_6", "machine_6_task_7", "machine_6_task_8", "machine_6_task_9", "machine_6_task_10", "machine_6_task_11", "machine_6_task_12"],
    ["machine_7_task_1", "machine_7_task_2", "machine_7_task_3", "machine_7_task_4", "machine_7_task_5", "machine_7_task_6", "machine_7_task_7", "machine_7_task_8", "machine_7_task_9", "machine_7_task_10", "machine_7_task_11", "machine_7_task_12"],
    ["machine_8_task_1", "machine_8_task_2", "machine_8_task_3", "machine_8_task_4", "machine_8_task_5", "machine_8_task_6", "machine_8_task_7", "machine_8_task_8", "machine_8_task_9", "machine_8_task_10", "machine_8_task_11", "machine_8_task_12"],
    ["machine_9_task_1", "machine_9_task_2", "machine_9_task_3", "machine_9_task_4", "machine_9_task_5", "machine_9_task_6", "machine_9_task_7", "machine_9_task_8", "machine_9_task_9", "machine_9_task_10", "machine_9_task_11", "machine_9_task_12"],
    ["machine_10_task_1", "machine_10_task_2", "machine_10_task_3", "machine_10_task_4", "machine_10_task_5", "machine_10_task_6", "machine_10_task_7", "machine_10_task_8", "machine_10_task_9", "machine_10_task_10", "machine_10_task_11", "machine_10_task_12"],
    ["machine_11_task_1", "machine_11_task_2", "machine_11_task_3", "machine_11_task_4", "machine_11_task_5", "machine_11_task_6", "machine_11_task_7", "machine_11_task_8", "machine_11_task_9", "machine_11_task_10", "machine_11_task_11", "machine_11_task_12"],
    ["machine_12_task_1", "machine_12_task_2", "machine_12_task_3", "machine_12_task_4", "machine_12_task_5", "machine_12_task_6", "machine_12_task_7", "machine_12_task_8", "machine_12_task_9", "machine_12_task_10", "machine_12_task_11", "machine_12_task_12"]
  ],
  "machines": [
    "machine_1", "machine_2", "machine_3", "machine_4", "machine_5", "machine_6", "machine_7", "machine_8", "machine_9", "machine_10", "machine_11", "machine_12"
  ],
  "tasks": [
    "task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7", "task_8", "task_9", "task_10", "task_11", "task_12"
  ]
}

Where $c_{ij}$ is the cost of assigning machine $i$ to task $j$, as specified in the retrieved cost matrix above. $x_{ij}$ is a binary variable equal to 1 if machine $i$ is assigned to task $j$, and 0 otherwise.