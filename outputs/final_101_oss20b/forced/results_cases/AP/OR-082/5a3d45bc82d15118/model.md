##### Objective Function: 

$ \min \sum_{i \in N} \sum_{j \in N} d_{ij} x_{ij} $

##### Constraints

###### 1. Depot Outgoing and Incoming Constraints: 

$ \sum_{j \in N,\, j \neq Depot} x_{Depot,j} = 1 $

$ \sum_{i \in N,\, i \neq Depot} x_{i,Depot} = 1 $

###### 2. Customer Visit Constraints: 

For each $k \in \{A,B,C\}$

$ \sum_{j \in N,\, j \neq k} x_{k,j} = 1 $

$ \sum_{i \in N,\, i \neq k} x_{i,k} = 1 $

###### 3. Subtour Elimination (MTZ) Constraints: 

For all $i,j \in \{A,B,C\}$ with $i \neq j$

$ u_i - u_j + 4\,x_{ij} \le 3 $

###### 4. Variable Constraints: 

$ x_{ij} \in \{0,1\} \quad \forall i,j \in N,\; i \neq j $

$ 2 \le u_i \le 4 \quad \forall i \in \{A,B,C\} $

##### Retrieved Information

{
  "distance_matrix": {
    "Depot": {"Depot":0, "A":28, "B":41, "C":63},
    "A": {"Depot":28, "A":0, "B":27, "C":87},
    "B": {"Depot":41, "A":27, "B":0, "C":81},
    "C": {"Depot":63, "A":87, "B":81, "C":0}
  },
  "nodes": ["Depot","A","B","C"],
  "customers": ["A","B","C"]
}