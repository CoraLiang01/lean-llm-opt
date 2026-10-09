### Mathematical Model

Let $I$ be the set of products from file_1_view_0[ProductName].

**Parameters:**
- $v_i$: value per unit of product $i$ (file_1_view_0[Value])
- $w_i$: weight per unit of product $i$ (file_1_view_0[Weight])
- $C$: total stock capacity (file_0_view_0[Capacity])

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of product $i$ to order each day

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0,\quad x_i \in \mathbb{Z},\quad \forall i \in I
\]

---

### Data Mapping

- $I$: file_1_view_0[ProductName]
- $v_i$: file_1_view_0[Value], keyed by ProductName
- $w_i$: file_1_view_0[Weight], keyed by ProductName
- $C$: file_0_view_0[Capacity]
- $x_i$: number of units of product $i$ to order each day (decision variable, indexed by file_1_view_0[ProductName])