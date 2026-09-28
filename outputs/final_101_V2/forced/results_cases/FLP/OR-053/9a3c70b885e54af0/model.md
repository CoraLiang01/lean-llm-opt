##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$, for each shelf $i \in S$ and product $j \in P$.

##### Parameters

- $S = \{1,2,3,4,5,6,7,8,9,10\}$ (Shelf IDs)
- $P = \{1,2,3,\ldots,20\}$ (Product Names)
- Shelf capacities:
  - $C_1 = 500$
  - $C_2 = 700$
  - $C_3 = 600$
  - $C_4 = 800$
  - $C_5 = 550$
  - $C_6 = 900$
  - $C_7 = 650$
  - $C_8 = 750$
  - $C_9 = 820$
  - $C_{10} = 570$
- Product values and weights:
  - $v_1 = 50$, $w_1 = 10$
  - $v_2 = 70$, $w_2 = 20$
  - $v_3 = 30$, $w_3 = 5$
  - $v_4 = 60$, $w_4 = 15$
  - $v_5 = 80$, $w_5 = 25$
  - $v_6 = 90$, $w_6 = 30$
  - $v_7 = 40$, $w_7 = 12$
  - $v_8 = 100$, $w_8 = 35$
  - $v_9 = 55$, $w_9 = 10$
  - $v_{10} = 75$, $w_{10} = 20$
  - $v_{11} = 65$, $w_{11} = 18$
  - $v_{12} = 95$, $w_{12} = 28$
  - $v_{13} = 45$, $w_{13} = 8$
  - $v_{14} = 85$, $w_{14} = 22$
  - $v_{15} = 70$, $w_{15} = 25$
  - $v_{16} = 110$, $w_{16} = 40$
  - $v_{17} = 50$, $w_{17} = 14$
  - $v_{18} = 60$, $w_{18} = 16$
  - $v_{19} = 120$, $w_{19} = 50$
  - $v_{20} = 100$, $w_{20} = 30$

##### Objective Function

$\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}$

##### Constraints

1. Shelf capacity constraints:
   $$
   \sum_{j \in P} w_j x_{ij} \leq C_i, \quad \forall i \in S
   $$
2. Integer and nonnegativity constraints:
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S,\, j \in P
   $$

##### Full Model

Let $S = \{1,2,3,4,5,6,7,8,9,10\}$, $P = \{1,2,\ldots,20\}$.

$\displaystyle \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}$

Subject to:

$\displaystyle \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,10$

$x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10,\ j=1,\ldots,20$

Where:

- $C = [500, 700, 600, 800, 550, 900, 650, 750, 820, 570]$
- $v = [50, 70, 30, 60, 80, 90, 40, 100, 55, 75, 65, 95, 45, 85, 70, 110, 50, 60, 120, 100]$
- $w = [10, 20, 5, 15, 25, 30, 12, 35, 10, 20, 18, 28, 8, 22, 25, 40, 14, 16, 50, 30]$