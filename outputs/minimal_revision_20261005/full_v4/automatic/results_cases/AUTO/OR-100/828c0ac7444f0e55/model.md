**Abstract Mathematical Model**

**Index Sets:**
- $I$: set of components, indexed by $i$ (from all Unnamed: 0 in file_1_view_0 and columns C1–C111 in file_0_view_0)
- $K$: set of workshops, indexed by $k$ (from all workshop in file_2_view_0 and Unnamed: 0 in file_0_view_0)

**Parameters:**
- $p_i$: unit price of component $i$  
  (from unit_price in file_1_view_0, indexed by Unnamed: 0)
- $a_{ki}$: unit processing time of component $i$ in workshop $k$  
  (from file_0_view_0, row Unnamed: 0 = $k$, column $i$)
- $b_k$: total available working hours in workshop $k$  
  (from total_hours in file_2_view_0, indexed by workshop)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of component $i$ to produce

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} a_{ki} x_i \leq b_k, \quad \forall k \in K
\]
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: All Unnamed: 0 in file_1_view_0 and columns C1–C111 in file_0_view_0
- $K$: All workshop in file_2_view_0 and Unnamed: 0 in file_0_view_0
- $p_i$: file_1_view_0, columns: Unnamed: 0 (component ID), unit_price
- $a_{ki}$: file_0_view_0, rows: Unnamed: 0 (workshop), columns: C1–C111 (component ID)
- $b_k$: file_2_view_0, columns: workshop, total_hours

**Variable Domain**

- $x_i$: integer, $x_i \geq 0$ for all $i \in I$