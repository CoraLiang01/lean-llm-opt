Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- For each shelf (from capacity.csv):
    - resource_id $i$ ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    - resource_capacity $c_i$:
        - $c_1 = 500$
        - $c_2 = 700$
        - $c_3 = 600$
        - $c_4 = 800$
        - $c_5 = 550$
        - $c_6 = 900$
        - $c_7 = 650$
        - $c_8 = 750$
        - $c_9 = 820$
        - $c_{10} = 570$
- For each product (from products.csv):
    - item_name $j$ ∈ {1, 2, ..., 20}
    - item_value $v_j$ and resource_requirement $w_j$:
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

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{1,\ldots,20\}} v_j \cdot x_{ij}
\]

**Subject to:**

- **Shelf capacity constraints (for each shelf $i$):**
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

- **Integrality and nonnegativity:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\; j \in \{1,\ldots,20\}
\]

---

**Where:**

- $x_{ij}$ = number of units of product $j$ (item_name $j$) placed on shelf $i$ (resource_id $i$)
- $v_j$ = item_value of product $j$
- $w_j$ = resource_requirement (weight) of product $j$
- $c_i$ = resource_capacity of shelf $i$

**All coefficients and identifiers are as retrieved and preserved in source order.**