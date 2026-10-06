#### Abstract Mathematical Model

Let:
- $I$ = set of produce types, indexed by $i$ (from file_1_view_0, column ProductName)
- For each $i \in I$:
    - $v_i$ = value (benefit) per unit of produce $i$ (file_1_view_0, column Value)
    - $w_i$ = weight per unit of produce $i$ (file_1_view_0, column Weight)
- $C$ = overall inventory capacity (file_0_view_0, column Capacity)
- $x_i$ = number of units of produce $i$ to order daily (decision variable, integer, $x_i \geq 0$)

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$ (produce types): file_1_view_0, column ProductName
- $v_i$ (benefit per unit): file_1_view_0, column Value, keyed by ProductName
- $w_i$ (weight per unit): file_1_view_0, column Weight, keyed by ProductName
- $C$ (overall capacity): file_0_view_0, column Capacity
- $x_i$ (order quantity): decision variable for each $i \in I$