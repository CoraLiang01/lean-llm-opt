##### Abstract Model

**Index Sets**  
$\,I\,$ – set of platforms  
$\,J\,$ – set of game genres  

**Parameters**  
$\,\text{capacity}_i\,$ – memory capacity of platform $i$, $i\in I$  
$\,\text{value}_j\,$ – value per unit of genre $j$, $j\in J$  
$\,\text{memory}_j\,$ – memory requirement per unit of genre $j$, $j\in J$  

**Decision Variables**  
$\,x_{ij}\,$ – integer number of units of games from genre $j$ to be listed on platform $i$, $i\in I,\;j\in J$  

**Objective Function**  
$\quad \max \quad \sum_{i\in I}\sum_{j\in J} \text{value}_j \cdot x_{ij}$  

**Constraints**  

1. **Memory Capacity Constraints**  
$\quad \sum_{j\in J} \text{memory}_j \cdot x_{ij} \;\le\; \text{capacity}_i , \quad \forall i\in I$  

2. **Variable Domain Constraints**  
$\quad x_{ij}\in \mathbb{Z}, \; x_{ij}\;\ge\;0 , \quad \forall i\in I,\; j\in J$  

##### Data Mapping

| Source Table | Exact Column Names | Aggregation for Parameters |
|--------------|--------------------|----------------------------|
| `file_0_view_0` | `PlatformId`, `Capacity` | `capacity_i` is taken directly from the `Capacity` column for each `PlatformId`. |
| `file_1_view_0` | `ProductName`, `Value`, `Weight` | For each genre `j`, `value_j` is taken from the `Value` column for the corresponding `ProductName`, and `memory_j` is taken from the `Weight` column for the corresponding `ProductName`. |