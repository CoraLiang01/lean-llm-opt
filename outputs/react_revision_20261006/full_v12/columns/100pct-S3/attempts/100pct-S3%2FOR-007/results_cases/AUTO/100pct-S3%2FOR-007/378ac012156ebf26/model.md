## Mathematical Model

Let $I$ be the set of vehicle types (indexed by $i$), with each $i$ corresponding to a unique ProductName from file_1_view_0[ProductName].

**Parameters:**
- $p_i$: profit per unit of vehicle $i$ (file_1_view_0[Value])
- $w_i$: inventory space required per unit of vehicle $i$ (file_1_view_0[Weight])
- $C$: total inventory capacity (file_0_view_0[Capacity])

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: number of vehicles of type $i$ to order per day

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

## Data Mapping

- $I$: All records in file_1_view_0[ProductName]
- $p_i$: file_1_view_0[Value], mapped by ProductName
- $w_i$: file_1_view_0[Weight], mapped by ProductName
- $C$: file_0_view_0[Capacity] (single value)
- $x_i$: decision variable for each $i \in I$ (ProductName from file_1_view_0)

All parameters and index sets are defined directly from the current CSV data.