##### Objective Function:

$\quad \min \sum_{i=1}^{15} \sum_{j=1}^{15} d_{ij} x_{ij}$

##### Constraints

###### 1. Tour Constraints (each location is entered and left exactly once):

$\sum_{j=1,\, j \ne i}^{15} x_{ij} = 1 \quad \forall i \in \{1,\ldots,15\}$

$\sum_{i=1,\, i \ne j}^{15} x_{ij} = 1 \quad \forall j \in \{1,\ldots,15\}$

###### 2. Subtour Elimination Constraints (MTZ formulation):

$u_1 = 1$

$2 \leq u_i \leq 15 \quad \forall i \in \{2,\ldots,15\}$

$u_i - u_j + 15\, x_{ij} \leq 14 \quad \forall i \in \{2,\ldots,15\},\, j \in \{2,\ldots,15\},\, i \ne j$

###### 3. Variable Domains:

$x_{ij} \in \{0,1\} \quad \forall i,j \in \{1,\ldots,15\},\, i \ne j$

$u_i \in \mathbb{Z} \quad \forall i \in \{1,\ldots,15\}$

##### Data Mapping

- $d_{ij}$: Distance from location $i$ to location $j$, as given in table_id "file_0_view_0", with rows and columns corresponding to locations $1$ through $15$ (row $i$ to column $j$).
- $x_{ij}$: Binary variable, $1$ if the tour goes directly from location $i$ to location $j$, $0$ otherwise.
- $u_i$: Integer variable for subtour elimination (MTZ), representing the position of location $i$ in the tour.

- The tour starts and ends at location $1$.

- All indices $i, j$ run over the set $\{1,2,\ldots,15\}$, corresponding to the 15 locations/customers.

- The distance matrix is symmetric and is sourced from table_id "file_0_view_0" in "20.csv".