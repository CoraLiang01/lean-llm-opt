Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$, where $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv).

Parameters:
- $v_j$: Value of product $j$ (from Value column in products.csv)
- $w_j$: Weight of product $j$ (from Weight column in products.csv)
- $C_i$: Capacity of shelf $i$ (from Capacity column in capacity.csv)

Data:
- Shelf capacities:
  - $C_1 = 500$, $C_2 = 700$, $C_3 = 600$, $C_4 = 800$, $C_5 = 550$, $C_6 = 900$, $C_7 = 650$, $C_8 = 750$, $C_9 = 820$, $C_{10} = 570$
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

Model:

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]

Subject to (for each shelf $i = 1,\ldots,10$):
\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\]

Where:
- $x_{ij}$: Number of units of product $j$ placed on shelf $i$ (integer, $\geq 0$)
- $v_j$: Value of product $j$ (see above)
- $w_j$: Weight of product $j$ (see above)
- $C_i$: Capacity of shelf $i$ (see above)