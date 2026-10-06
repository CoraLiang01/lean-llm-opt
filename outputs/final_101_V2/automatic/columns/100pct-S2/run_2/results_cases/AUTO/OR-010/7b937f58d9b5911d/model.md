Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

**Parameters:**

- Sections (from capacity.csv):  
  SectionID $\in \{1,2,3,4,5,6,7,8\}$  
  Section capacities $C_i$:
    - $C_1 = 100$
    - $C_2 = 150$
    - $C_3 = 120$
    - $C_4 = 130$
    - $C_5 = 90$
    - $C_6 = 110$
    - $C_7 = 160$
    - $C_8 = 140$

- Products (from products.csv):  
  ProductName $\in \{1,2,3,4,5,6,7,8,9,10\}$  
  Product values $p_j$:
    - $p_1 = 10$
    - $p_2 = 15$
    - $p_3 = 8$
    - $p_4 = 12$
    - $p_5 = 20$
    - $p_6 = 25$
    - $p_7 = 5$
    - $p_8 = 30$
    - $p_9 = 18$
    - $p_{10} = 22$
  Product space requirements $w_j$:
    - $w_1 = 2$
    - $w_2 = 3$
    - $w_3 = 1$
    - $w_4 = 2$
    - $w_5 = 4$
    - $w_6 = 5$
    - $w_7 = 1$
    - $w_8 = 6$
    - $w_9 = 3$
    - $w_{10} = 4$

**Decision Variables:**

- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in \{1,\ldots,8\}$, $j \in \{1,\ldots,10\}$

---

**Objective:**

$$
\max \sum_{i=1}^8 \sum_{j=1}^{10} p_j x_{ij}
$$

**Subject to:**

For each section $i$:
$$
\sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

---

**Data Used:**

- SectionID and Capacity from capacity.csv (rows 1–8, in file order)
- ProductName, Value, and Weight from products.csv (rows 1–10, in file order)
- All variables and constraints indexed by these explicit identifiers and coefficients.