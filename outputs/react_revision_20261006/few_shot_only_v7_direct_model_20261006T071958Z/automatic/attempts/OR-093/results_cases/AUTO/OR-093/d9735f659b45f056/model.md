##### Objective Function:

$\quad \min \sum_{i \in \text{Machines}} \sum_{j \in \text{Tasks}} c_{ij} \, x_{ij}$

##### Constraints:

1. **Assignment Constraints:**

$\sum_{j \in \text{Tasks}} x_{ij} = 1 \quad \forall i \in \text{Machines}$

$\sum_{i \in \text{Machines}} x_{ij} = 1 \quad \forall j \in \text{Tasks}$

2. **Variable Domains:**

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Machines}, \; j \in \text{Tasks}$

##### Data Mapping

- $\text{Machines} = $ all values in column "Machine" of table_id "file_0_view_0"
- $\text{Tasks} = $ all column names except "Machine" in table_id "file_0_view_0"
- $c_{ij} = $ value in row with "Machine" $=i$, column $=j$ of table_id "file_0_view_0" from source "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv"
- $x_{ij}$ is a binary decision variable indicating assignment of machine $i$ to task $j$