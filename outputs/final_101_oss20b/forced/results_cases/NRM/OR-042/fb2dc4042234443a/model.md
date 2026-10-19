**Abstract Optimization Model**

| Symbol | Meaning |
|--------|---------|
| \(I\) | Set of drug types (indexed by \(i\)) |
| \(v_i\) | Benefit coefficient of drug type \(i\) (from *products.csv*) |
| \(w_i\) | Weight of one unit of drug type \(i\) (from *products.csv*) |
| \(C\) | Total available capacity (from *capacity.csv*) |
| \(x_i\) | Integer number of units of drug type \(i\) to order daily |

---

### Model

**Objective**

\[
\max \sum_{i\in I} v_i \, x_i
\]

**Constraints**

1. **Capacity Constraint**

\[
\sum_{i\in I} w_i \, x_i \;\le\; C
\]

2. **Non‑negativity and Integer Domain**

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall i\in I
\]

---

### Data Mapping

| Table ID | Column(s) Used | Source |
|----------|----------------|--------|
| `file_0_view_0` | `Capacity` | *capacity.csv* |
| `file_1_view_0` | `ProductName`, `Value`, `Weight` | *products.csv* |

*No record values or counts are reproduced; only the exact table identifiers and column names are listed.*