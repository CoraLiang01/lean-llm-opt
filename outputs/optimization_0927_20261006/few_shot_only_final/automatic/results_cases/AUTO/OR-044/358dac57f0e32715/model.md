Let  
- $x_{ij}$ = number of units of product $j$ (ProductName $j$ from products.csv) to be placed in section $i$ (SectionID $i$ from capacity.csv), for all $i \in \{1,2,3,4,5,6,7,8\}$ and $j \in \{1,2,3,4,5,6,7,8,9,10\}$.

Parameters:  
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight (space requirement) of product $j$ (from products.csv)
- $C_i$ = Capacity of section $i$ (from capacity.csv)

Data:

From capacity.csv:  
| SectionID | Capacity |
|-----------|----------|
| 1         | 100      |
| 2         | 150      |
| 3         | 120      |
| 4         | 130      |
| 5         | 90       |
| 6         | 110      |
| 7         | 160      |
| 8         | 140      |

From products.csv:  
| ProductName | Value | Weight |
|-------------|-------|--------|
| 1           | 10    | 2      |
| 2           | 15    | 3      |
| 3           | 8     | 1      |
| 4           | 12    | 2      |
| 5           | 20    | 4      |
| 6           | 25    | 5      |
| 7           | 5     | 1      |
| 8           | 30    | 6      |
| 9           | 18    | 3      |
| 10          | 22    | 4      |

Model:

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

**Subject to:**

Section capacity constraints (for each section $i$):
$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \quad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,3,4,5,6,7,8\},\; j \in \{1,2,3,4,5,6,7,8,9,10\}
$$

Where:
- $v_j$ and $w_j$ are as given in products.csv for ProductName $j$
- $C_i$ is as given in capacity.csv for SectionID $i$