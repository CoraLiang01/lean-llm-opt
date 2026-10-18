##### Sets and Indices

Let $N = \{\text{Depot}, A, B, C\}$ denote the set of nodes (locations), indexed by $i$ and $j$.

##### Parameters

Let $d_{ij}$ denote the distance from node $i$ to node $j$, for all $i, j \in N$.

The retrieved distance matrix (restricted to the relevant nodes) is:

|         | Depot |   A   |   B   |   C   |
|---------|-------|-------|-------|-------|
| Depot   |   0   |  28   |  41   |  63   |
| A       |  28   |   0   |  27   |  87   |
| B       |  41   |  27   |   0   |  81   |
| C       |  63   |  87   |  81   |   0   |

So,
- $d_{\text{Depot},A} = 28$, $d_{\text{Depot},B} = 41$, $d_{\text{Depot},C} = 63$
- $d_{A,\text{Depot}} = 28$, $d_{A,B} = 27$, $d_{A,C} = 87$
- $d_{B,\text{Depot}} = 41$, $d_{B,A} = 27$, $d_{B,C} = 81$
- $d_{C,\text{Depot}} = 63$, $d_{C,A} = 87$, $d_{C,B} = 81$

##### Decision Variables

Let
- $x_{ij} = \begin{cases} 1 & \text{if the route goes directly from node } i \text{ to node } j \\ 0 & \text{otherwise} \end{cases}$, for all $i, j \in N$, $i \neq j$

Let
- $u_i$ be the position of node $i$ in the tour (for subtour elimination), for all $i \in N$, $i \neq \text{Depot}$

##### Objective Function

$\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}$

##### Constraints

1. **Departure and Arrival Constraints:**
   - The van must leave each node exactly once:
     $$
     \sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N
     $$
   - The van must arrive at each node exactly once:
     $$
     \sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N
     $$

2. **Subtour Elimination Constraints (MTZ Formulation):**
   - For all $i, j \in N$, $i \neq j$, $i \neq \text{Depot}$, $j \neq \text{Depot}$:
     $$
     u_i - u_j + 3 x_{ij} \leq 2
     $$
   - $u_i \in \{1,2,3\}$ for $i \in \{A, B, C\}$

3. **Variable Domains:**
   - $x_{ij} \in \{0,1\}$ for all $i, j \in N$, $i \neq j$

##### Retrieved Information

{
  "nodes": ["Depot", "A", "B", "C"],
  "distance": {
    "Depot": {"Depot": 0, "A": 28, "B": 41, "C": 63},
    "A": {"Depot": 28, "A": 0, "B": 27, "C": 87},
    "B": {"Depot": 41, "A": 27, "B": 0, "C": 81},
    "C": {"Depot": 63, "A": 87, "B": 81, "C": 0}
  }
}

##### Full Mathematical Model

$\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}$

Subject to:
- $\sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \quad \forall i \in N$
- $\sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \quad \forall j \in N$
- $u_i - u_j + 3 x_{ij} \leq 2 \quad \forall i, j \in \{A, B, C\},\ i \neq j$
- $u_i \in \{1,2,3\} \quad \forall i \in \{A, B, C\}$
- $x_{ij} \in \{0,1\} \quad \forall i, j \in N,\ i \neq j$