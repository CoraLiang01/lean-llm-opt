##### Objective Function: 

$\quad \quad \min \sum_{i \in \{Depot,A,B,C\}} \sum_{j \in \{Depot,A,B,C\},\, j \neq i} d_{ij} \cdot x_{ij}$

##### Constraints

###### 1. Out‑degree Constraints: 

$\quad \quad \sum_{j \in \{Depot,A,B,C\},\, j \neq i} x_{ij} = 1, \quad \forall i \in \{Depot,A,B,C\}$

###### 2. In‑degree Constraints: 

$\quad \quad \sum_{i \in \{Depot,A,B,C\},\, i \neq j} x_{ij} = 1, \quad \forall j \in \{Depot,A,B,C\}$

###### 3. Subtour Elimination (MTZ) Constraints: 

Let $u_{i}$ be a continuous variable for $i \in \{A,B,C\}$ with $u_{Depot}=0$.

$\quad \quad u_{i} - u_{j} + 3 \cdot x_{ij} \leq 2, \quad \forall i,j \in \{A,B,C\},\, i \neq j$

###### 4. Variable Domains: 

$\quad \quad x_{ij} \in \{0,1\}, \quad \forall i,j \in \{Depot,A,B,C\},\, i \neq j$

$\quad \quad u_{i} \geq 0, \quad \forall i \in \{A,B,C\}$

##### Retrieved Information

{
  "nodes": ["Depot", "A", "B", "C"],
  "distances": {
    "Depot-A": 28,
    "Depot-B": 41,
    "Depot-C": 63,
    "A-Depot": 28,
    "A-B": 27,
    "A-C": 87,
    "B-Depot": 41,
    "B-A": 27,
    "B-C": 81,
    "C-Depot": 63,
    "C-A": 87,
    "C-B": 81
  }
}

##### Optimal Sequence(s)

The optimal total travel distance is **199 km**, achieved by either of the following routes:

1. Depot → A → B → C → Depot  
2. Depot → C → B → A → Depot