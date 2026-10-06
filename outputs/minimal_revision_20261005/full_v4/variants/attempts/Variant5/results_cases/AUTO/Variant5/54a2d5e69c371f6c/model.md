##### Objective Function:

$\quad \min \sum_{i \in \mathcal{M}} \sum_{j \in \mathcal{J}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints (Each job assigned to exactly one team):

$\sum_{i \in \mathcal{M}} x_{ij} = 1 \quad \forall j \in \mathcal{J}$

###### 2. Capacity Constraints (Team capacity not exceeded):

$\sum_{j \in \mathcal{J}} r_{ij} \, x_{ij} \leq K_i \quad \forall i \in \mathcal{M}$

###### 3. Variable Constraints (Binary assignments):

$x_{ij} \in \{0,1\} \quad \forall i \in \mathcal{M}, \; j \in \mathcal{J}$

---

##### Index Sets

- $\mathcal{M}$: Set of teams (Machines) from column "Machine" in `machine_capacity.csv`, `assignment_costs.csv`, and `assignment_resources.csv`.
- $\mathcal{J}$: Set of jobs $\{ \text{J1}, \text{J2}, \ldots, \text{J8} \}$ from columns in `assignment_costs.csv` and `assignment_resources.csv`.

##### Parameters

- $K_i$: Capacity of team $i$ from column "Capacity" in `machine_capacity.csv`.
- $c_{ij}$: Cost of assigning job $j$ to team $i$ from table `assignment_costs.csv`, row "Machine" $i$, column $j$.
- $r_{ij}$: Capacity consumed if job $j$ is assigned to team $i$ from table `assignment_resources.csv`, row "Machine" $i$, column $j$.

##### Variables

- $x_{ij}$: Binary variable, $1$ if job $j$ is assigned to team $i$, $0$ otherwise.

---

##### Data Mapping

```json
{
  "teams": {
    "table_id": "file_0_view_0",
    "column": "Machine"
  },
  "jobs": [
    "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8"
  ],
  "capacity": {
    "table_id": "file_0_view_0",
    "row_id": "Machine",
    "column": "Capacity"
  },
  "cost": {
    "table_id": "file_1_view_0",
    "row_id": "Machine",
    "column": ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8"]
  },
  "resource": {
    "table_id": "file_2_view_0",
    "row_id": "Machine",
    "column": ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8"]
  }
}
```