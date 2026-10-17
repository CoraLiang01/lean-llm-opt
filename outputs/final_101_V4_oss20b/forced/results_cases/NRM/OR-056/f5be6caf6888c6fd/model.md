**Mathematical Model**

Let  

- \(I\) be the set of display areas (indexed by `DisplayID`).  
- \(J\) be the set of vessel types (indexed by `ProductName`).  

Parameters (all taken directly from the source tables)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(c_i\) | Capacity of display area \(i\) | `file_0_view_0` – column `Capacity` |
| \(v_j\) | Value of vessel type \(j\) | `file_1_view_0` – column `Value` |
| \(w_j\) | Weight (size) of vessel type \(j\) | `file_1_view_0` – column `Weight` |

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\quad \forall i\in I,\; j\in J
\]

Objective (maximize total value)

\[
\max \;\; \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

Subject to (capacity of each display area)

\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; c_i \qquad \forall i\in I
\]

and non‑negativity

\[
x_{ij}\;\ge\;0 \qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(c_i\) | `file_0_view_0` | `Capacity` |
| \(v_j\) | `file_1_view_0` | `Value` |
| \(w_j\) | `file_1_view_0` | `Weight` |

All indices \(i\) and \(j\) are taken from the distinct values of `DisplayID` and `ProductName` in the respective tables. No additional data or constraints are introduced beyond those specified in the query.