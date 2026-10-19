**Abstract Optimization Model**

**Index Set**  
\(I\) – set of produce types (indexed by \(i\))

**Parameters**  
\(v_i\) – benefit (value) per unit of produce type \(i\)  
\(w_i\) – weight per unit of produce type \(i\)  
\(C\) – total available inventory capacity

**Decision Variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of produce type \(i\) to order daily

**Objective Function**  
\[
\max \sum_{i \in I} v_i \, x_i
\]

**Constraints**  

1. **Capacity Constraint**  
\[
\sum_{i \in I} w_i \, x_i \;\le\; C
\]

2. **Non‑negativity and Integer Domain**  
\[
x_i \;\in\; \mathbb{Z}_{\ge 0}, \qquad \forall i \in I
\]

---

### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(v_i\)   | `file_1_view_0` (`products.csv`) | `Value` |
| \(w_i\)   | `file_1_view_0` (`products.csv`) | `Weight` |
| \(C\)     | `file_0_view_0` (`capacity.csv`) | `Capacity` |

*All table identifiers (`file_0_view_0`, `file_1_view_0`) and column names are taken directly from the provided CSV metadata.*