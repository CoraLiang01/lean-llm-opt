Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- For each shelf $i$ (resource_id): resource_capacity $c_i$
- For each product $j$ (item_name): item_value $v_j$, resource_requirement $a_j$

**Data:**

- Shelves (resource_id, resource_capacity):

    1: 500  
    2: 700  
    3: 600  
    4: 800  
    5: 550  
    6: 900  
    7: 650  
    8: 750  
    9: 820  
    10: 570  

- Products (item_name, item_value, resource_requirement):

    1: 50, 10  
    2: 70, 20  
    3: 30, 5  
    4: 60, 15  
    5: 80, 25  
    6: 90, 30  
    7: 40, 12  
    8: 100, 35  
    9: 55, 10  
    10: 75, 20  
    11: 65, 18  
    12: 95, 28  
    13: 45, 8  
    14: 85, 22  
    15: 70, 25  
    16: 110, 40  
    17: 50, 14  
    18: 60, 16  
    19: 120, 50  
    20: 100, 30  

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

where $v_j$ is the item_value for product $j$.

**Constraints:**

For each shelf $i$ (resource_id):

$$
\sum_{j=1}^{20} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $a_j$ is the resource_requirement for product $j$, and $c_i$ is the resource_capacity for shelf $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

**Explicitly, for each shelf $i$ (resource_id):**

For $i=1$ (resource_id 1, resource_capacity 500):

$$
10x_{1,1} + 20x_{1,2} + 5x_{1,3} + 15x_{1,4} + 25x_{1,5} + 30x_{1,6} + 12x_{1,7} + 35x_{1,8} + 10x_{1,9} + 20x_{1,10} + 18x_{1,11} + 28x_{1,12} + 8x_{1,13} + 22x_{1,14} + 25x_{1,15} + 40x_{1,16} + 14x_{1,17} + 16x_{1,18} + 50x_{1,19} + 30x_{1,20} \leq 500
$$

Repeat similarly for $i=2$ to $i=10$ with their respective resource_capacity.

---

**Summary:**

- Maximize total value of products allocated to shelves.
- For each shelf, total weight of allocated products cannot exceed its capacity.
- All allocations are nonnegative integers.