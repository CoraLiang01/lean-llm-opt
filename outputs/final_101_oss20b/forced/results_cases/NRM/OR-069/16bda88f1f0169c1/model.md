**Abstract Assignment Model**

---

### Index Sets  
- **M** – set of managers.  
- **P** – set of projects.  

### Parameters  
- **c<sub>ij</sub>** – cost of assigning manager *i* ∈ *M* to project *j* ∈ *P*.  

### Decision Variables  
- **x<sub>ij</sub>** ∈ {0, 1} – binary variable that equals 1 if manager *i* is assigned to project *j*, and 0 otherwise.  

### Objective Function  
Minimize the total assignment cost:  
$$\min \sum_{i \in M} \sum_{j \in P} c_{ij}\,x_{ij}$$  

### Constraints  

1. **Manager assignment constraints** – each manager is assigned to exactly one project:  
   $$\sum_{j \in P} x_{ij} = 1,\quad \forall i \in M$$  

2. **Project assignment constraints** – each project is assigned to exactly one manager:  
   $$\sum_{i \in M} x_{ij} = 1,\quad \forall j \in P$$  

3. **Variable domain constraints** – binary decision variables:  
   $$x_{ij} \in \{0,1\},\quad \forall i \in M,\; \forall j \in P$$  

### Data Mapping  

| Source | Table ID | Column Names |
|--------|----------|--------------|
| Cost matrix | `manager_project_costs.csv` | `Manager`, `Project 1 Cost`, `Project 2 Cost`, …, `Project 11 Cost` |

*The table `manager_project_costs.csv` contains one row per manager and one column per project cost. No record values or counts are reproduced here.*