**Mathematical Model**

Let  

- \(I=\{1,\dots,25\}\) be the set of vehicle types (index \(i\) corresponds to the \(i\)-th row of *products.csv*).  
- \(x_i\) = number of vehicles of type \(i\) ordered per day.  

Parameters (all taken directly from the supplied tables)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(p_i\) | Profit per vehicle of type \(i\) (Value) | *products.csv* |
| \(w_i\) | Weight (space) per vehicle of type \(i\) (Weight) | *products.csv* |
| \(C\) | Total inventory capacity (Capacity) | *capacity.csv* |

**Objective**

\[
\max \;\; Z = \sum_{i\in I} p_i\,x_i
\]

**Constraints**

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

\[
x_i \;\ge\; 0,\;\; x_i \in \mathbb{Z}\quad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) | Notes |
|-----------|----------|-----------|-------|
| \(C\) | `file_0_view_0` | `Capacity` | Single value row |
| \(p_i\) | `file_1_view_0` | `Value` | One value per product row |
| \(w_i\) | `file_1_view_0` | `Weight` | One value per product row |

All indices \(i\) are taken from the 25 rows returned in `file_1_view_0`. No additional filtering or aggregation is applied.