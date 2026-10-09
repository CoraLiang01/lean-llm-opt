##### Objective Function:

$\quad \min \sum_{i \in \text{Machines}} \sum_{j \in \text{Jobs}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints (each job assigned to exactly one team):

$\sum_{i \in \text{Machines}} x_{ij} = 1 \quad \forall j \in \text{Jobs}$

###### 2. Capacity Constraints (team capacity not exceeded):

$\sum_{j \in \text{Jobs}} r_{ij} \, x_{ij} \leq \text{Capacity}_i \quad \forall i \in \text{Machines}$

###### 3. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Machines}, \forall j \in \text{Jobs}$

##### Data Mapping

- $\text{Machines}$: All Machine values from file_0_view_0, file_1_view_0, and file_2_view_0 ("M1", "M2", "M3", "M4")
- $\text{Jobs}$: All job columns J1–J8 from file_1_view_0 and file_2_view_0
- $c_{ij}$: Assignment cost for team $i$ and job $j$ from file_1_view_0, column $j$, row with Machine $i$
- $r_{ij}$: Assignment resource usage for team $i$ and job $j$ from file_2_view_0, column $j$, row with Machine $i$
- $\text{Capacity}_i$: Capacity for team $i$ from file_0_view_0, column "Capacity", row with Machine $i$
- $x_{ij}$: Binary variable, 1 if job $j$ assigned to team $i$, 0 otherwise

- Table IDs:
    - file_0_view_0: machine_capacity.csv (columns: Machine, Capacity)
    - file_1_view_0: assignment_costs.csv (columns: Machine, J1, ..., J8)
    - file_2_view_0: assignment_resources.csv (columns: Machine, J1, ..., J8)