**Sets**  
Let $I$ be the set of areas, indexed by $i$.  
Let $I$ correspond to all ProductName values in file_1_view_0 (products.csv).

**Parameters**  
For each $i \in I$:
- $v_i$: benefit coefficient for area $i$ (Value column in file_1_view_0)
- $w_i$: development unit weight for area $i$ (Weight column in file_1_view_0)

Let $C$ be the overall development capacity (Capacity column in file_0_view_0).

**Decision Variables**  
For each $i \in I$:
- $x_i \in \mathbb{Z}_{\geq 0}$: integer scale of development in area $i$ per day

**Objective**  
Maximize total benefit:
$$
\max \sum_{i \in I} v_i x_i
$$

**Constraints**  
Overall development capacity:
$$
\sum_{i \in I} w_i x_i \leq C
$$

Integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

**Data Mapping**

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: file_1_view_0, column Value, key ProductName
- $w_i$: file_1_view_0, column Weight, key ProductName
- $C$: file_0_view_0, column Capacity

**Variables**

- $x_i$: integer, for each ProductName in file_1_view_0

**Objective and Constraints**

- Objective: maximize $\sum_{i \in I} v_i x_i$
- Subject to: $\sum_{i \in I} w_i x_i \leq C$
- $x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$