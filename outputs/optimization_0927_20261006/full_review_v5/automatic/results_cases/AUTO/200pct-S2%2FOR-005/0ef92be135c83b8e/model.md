##### Sets and Indices

Let $I$ be the set of bread types, indexed by $i$.

##### Parameters

For each bread type $i$ (see table below):

- $v_i$ = expected profit per unit of bread $i$ (from "item_value")
- $a_i$ = storage space required per unit of bread $i$ (from "resource_requirement")

Let $C$ = total storage capacity (from "resource_capacity" in capacity.csv)

##### Decision Variables

For each bread type $i$:

- $x_i$ = number of units of bread $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Objective Function

$$
\max \sum_{i \in I} v_i x_i
$$

##### Constraints

1. **Storage Capacity Constraint:**
   $$
   \sum_{i \in I} a_i x_i \leq C
   $$

2. **Integrality and Nonnegativity:**
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Parameter Table

| item_name        | item_value ($v_i$) | resource_requirement ($a_i$) |
|------------------|-------------------|------------------------------|
| Baguette         | 888               | 4                            |
| Croissant        | 134               | 2                            |
| Sourdough        | 129               | 4                            |
| Rye Bread        | 370               | 3                            |
| Brioche          | 921               | 2                            |
| Focaccia         | 765               | 1                            |
| Ciabatta         | 154               | 2                            |
| Pita             | 837               | 1                            |
| Bagel            | 584               | 3                            |
| English Muffin   | 365               | 3                            |

Total storage capacity: $C = 180$

---

#### Complete Model

$$
\begin{align*}
\max \quad & 888x_{\text{Baguette}} + 134x_{\text{Croissant}} + 129x_{\text{Sourdough}} + 370x_{\text{Rye Bread}} + 921x_{\text{Brioche}} \\
& + 765x_{\text{Focaccia}} + 154x_{\text{Ciabatta}} + 837x_{\text{Pita}} + 584x_{\text{Bagel}} + 365x_{\text{English Muffin}} \\
\text{s.t.} \quad & 4x_{\text{Baguette}} + 2x_{\text{Croissant}} + 4x_{\text{Sourdough}} + 3x_{\text{Rye Bread}} + 2x_{\text{Brioche}} \\
& + 1x_{\text{Focaccia}} + 2x_{\text{Ciabatta}} + 1x_{\text{Pita}} + 3x_{\text{Bagel}} + 3x_{\text{English Muffin}} \leq 180 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\end{align*}
$$