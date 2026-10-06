#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of components (from file_1_view_0.Unnamed: 0, e.g., C1, C2, ..., C111)
- $W$: set of workshops (from file_0_view_0.Unnamed: 0, e.g., Casting, Milling, Finishing, Assembly, QA & Packaging)

**Parameters:**
- $p_i$: unit price of component $i$ (from file_1_view_0.unit_price, key: Unnamed: 0)
- $a_{wi}$: unit processing time of component $i$ in workshop $w$ (from file_0_view_0, key: Unnamed: 0 for $w$, column $i$ for $i$)
- $c_w$: total available working hours in workshop $w$ (from file_2_view_0.total_hours, key: workshop)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units to produce of component $i$

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} a_{wi} x_i \leq c_w, \quad \forall w \in W
\]
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

---

#### Data Mapping

- $I$ (components): file_1_view_0.Unnamed: 0
- $W$ (workshops): file_0_view_0.Unnamed: 0 and file_2_view_0.workshop
- $p_i$: file_1_view_0.unit_price, key: Unnamed: 0
- $a_{wi}$: file_0_view_0, row with Unnamed: 0 = $w$, column $i$
- $c_w$: file_2_view_0.total_hours, key: workshop

---

**All parameters and sets are to be populated directly from the referenced columns and rows in the returned CSVQA data.**