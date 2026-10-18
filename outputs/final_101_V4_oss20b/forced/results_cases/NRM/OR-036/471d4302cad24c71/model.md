**Abstract Model**

| Symbol | Meaning |
|--------|---------|
| \(I\) | Index set of vehicle types |
| \(w_i\) | Benefit coefficient (value) of vehicle type \(i\) |
| \(c_i\) | Weight (inventory consumption) of vehicle type \(i\) |
| \(x_i\) | Integer number of units of vehicle type \(i\) ordered daily |
| \(C\) | Total inventory capacity |

---

### Model

**Objective**

\[
\max \sum_{i\in I} w_i \, x_i
\]

**Constraints**

1. **Capacity Constraint**

\[
\sum_{i\in I} c_i \, x_i \;\le\; C
\]

2. **Non‑negativity and Integer Domain**

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall i\in I
\]

---

### Data Mapping

| Table ID | Column(s) | Parameter |
|----------|-----------|-----------|
| `file_0_view_0` | `Capacity` | \(C\) |
| `file_1_view_0` | `ProductName` | Index set \(I\) (each distinct product name) |
| `file_1_view_0` | `Value` | \(w_i\) |
| `file_1_view_0` | `Weight` | \(c_i\) |

*No record values or counts are reproduced; only the source table identifiers and column names are listed.*