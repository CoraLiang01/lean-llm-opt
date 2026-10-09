##### Objective Function:

$\quad \min \sum_{i \in \text{Machines}} \sum_{j \in \text{Jobs}} c_{ij} \, x_{ij}$

##### Constraints:

1. **Job Assignment:**  
$\sum_{i \in \text{Machines}} x_{ij} = 1 \quad \forall j \in \text{Jobs}$

2. **Team Capacity:**  
$\sum_{j \in \text{Jobs}} r_{ij} \, x_{ij} \leq \text{Cap}_i \quad \forall i \in \text{Machines}$

3. **Binary Variables:**  
$x_{ij} \in \{0,1\} \quad \forall i \in \text{Machines}, \forall j \in \text{Jobs}$

##### Data Mapping

- $\text{Machines}$: All "Machine" values in `file_0_view_0` (machine_capacity.csv), `file_1_view_0` (assignment_costs.csv), and `file_2_view_0` (assignment_resources.csv).
- $\text{Jobs}$: All job columns J1–J8 in `file_1_view_0` and `file_2_view_0`.
- $c_{ij}$: Assignment cost for team $i$ and job $j$ from `file_1_view_0` (assignment_costs.csv), with row index "Machine" and column "J1"–"J8".
- $r_{ij}$: Assignment resource usage for team $i$ and job $j$ from `file_2_view_0` (assignment_resources.csv), with row index "Machine" and column "J1"–"J8".
- $\text{Cap}_i$: Capacity for team $i$ from `file_0_view_0` (machine_capacity.csv), column "Capacity", row "Machine".

All indices, parameters, and mappings are defined exactly as in the source tables.