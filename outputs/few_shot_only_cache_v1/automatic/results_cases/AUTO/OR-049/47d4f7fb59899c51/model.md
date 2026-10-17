Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let:
- $S$ = set of shelves, indexed by $i$ (from "capacity.csv", ShelfID: 1, 2, ..., 10)
- $P$ = set of products, indexed by $j$ (from "products.csv", ProductName: see below)
- $c_i$ = capacity of shelf $i$ (from "capacity.csv", Capacity)
- $v_j$ = value of product $j$ (from "products.csv", Value)
- $w_j$ = weight of product $j$ (from "products.csv", Weight)

#### Sets and Parameters

Shelves ($S$) and their capacities:
- Shelf 1: $c_1 = 5.0$
- Shelf 2: $c_2 = 7.0$
- Shelf 3: $c_3 = 6.0$
- Shelf 4: $c_4 = 8.0$
- Shelf 5: $c_5 = 5.5$
- Shelf 6: $c_6 = 9.0$
- Shelf 7: $c_7 = 6.5$
- Shelf 8: $c_8 = 7.5$
- Shelf 9: $c_9 = 8.2$
- Shelf 10: $c_{10} = 5.7$

Products ($P$) and their values/weights:
- Smartphone: $v_1 = 200$, $w_1 = 1.0$
- Laptop: $v_2 = 1500$, $w_2 = 5.0$
- Headphones: $v_3 = 100$, $w_3 = 0.5$
- Camera: $v_4 = 800$, $w_4 = 2.0$
- Smartwatch: $v_5 = 250$, $w_5 = 0.3$
- Tablet: $v_6 = 600$, $w_6 = 1.5$
- Bluetooth Speaker: $v_7 = 150$, $w_7 = 1.0$
- Keyboard: $v_8 = 80$, $w_8 = 0.8$
- Mouse: $v_9 = 50$, $w_9 = 0.2$
- Monitor: $v_{10} = 300$, $w_{10} = 3.0$
- Printer: $v_{11} = 400$, $w_{11} = 4.0$
- External Hard Drive: $v_{12} = 120$, $w_{12} = 0.5$
- Router: $v_{13} = 60$, $w_{13} = 0.3$
- Power Bank: $v_{14} = 40$, $w_{14} = 0.4$
- Memory Card: $v_{15} = 30$, $w_{15} = 0.05$
- USB Flash Drive: $v_{16} = 25$, $w_{16} = 0.02$
- Smart Home Hub: $v_{17} = 100$, $w_{17} = 0.6$
- Gaming Console: $v_{18} = 500$, $w_{18} = 4.0$
- Fitness Tracker: $v_{19} = 90$, $w_{19} = 0.2$
- E-Reader: $v_{20} = 180$, $w_{20} = 0.5$

#### Decision Variables

$$
x_{ij} = \text{number of units of product } j \text{ placed on shelf } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

#### Objective Function

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

#### Constraints

For each shelf $i \in S$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i
$$

For all $i \in S$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

#### Explicitly, with identifiers:

Let $S = \{1,2,3,4,5,6,7,8,9,10\}$ (ShelfID), $P$ = list of 20 products as above.

**Objective:**
$$
\max \sum_{i=1}^{10} \Bigg[
200\,x_{i,1} + 1500\,x_{i,2} + 100\,x_{i,3} + 800\,x_{i,4} + 250\,x_{i,5} + 600\,x_{i,6} + 150\,x_{i,7} + 80\,x_{i,8} + 50\,x_{i,9} + 300\,x_{i,10} + 400\,x_{i,11} + 120\,x_{i,12} + 60\,x_{i,13} + 40\,x_{i,14} + 30\,x_{i,15} + 25\,x_{i,16} + 100\,x_{i,17} + 500\,x_{i,18} + 90\,x_{i,19} + 180\,x_{i,20}
\Bigg]
$$

**For each shelf $i$ (1 to 10):**
$$
1.0\,x_{i,1} + 5.0\,x_{i,2} + 0.5\,x_{i,3} + 2.0\,x_{i,4} + 0.3\,x_{i,5} + 1.5\,x_{i,6} + 1.0\,x_{i,7} + 0.8\,x_{i,8} + 0.2\,x_{i,9} + 3.0\,x_{i,10} + 4.0\,x_{i,11} + 0.5\,x_{i,12} + 0.3\,x_{i,13} + 0.4\,x_{i,14} + 0.05\,x_{i,15} + 0.02\,x_{i,16} + 0.6\,x_{i,17} + 4.0\,x_{i,18} + 0.2\,x_{i,19} + 0.5\,x_{i,20} \leq c_i
$$
where $c_i$ is the capacity for shelf $i$ as listed above.

**Variable domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
$$