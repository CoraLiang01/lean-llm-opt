##### Abstract Model

**Index Set**  
\(I = \{\, i \mid \text{Product } i \text{ appears in table } \text{file\_0\_view\_0}\,\}\)

**Parameters**  
\[
\begin{aligned}
w_i & = \text{Revenue of product } i \quad (\text{from column } \text{Revenue})\\
a_i & = \text{Demand of product } i \quad (\text{from column } \text{Demand})\\
c_i & = \text{Initial Inventory of product } i \quad (\text{from column } \text{Initial Inventory})
\end{aligned}
\]

**Decision Variables**  
\[
x_i \;\in\; \mathbb{Z}_{+}\quad \forall i\in I
\]

**Objective Function**  
\[
\max \;\sum_{i\in I} w_i\,x_i
\]

**Constraints**

1. **Inventory Constraints**  
\[
x_i \;\le\; c_i \quad \forall i\in I
\]

2. **Demand Constraints**  
\[
x_i \;\le\; a_i \quad \forall i\in I
\]

---

##### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(w_i\)   | `file_0_view_0` | `Revenue` |
| \(a_i\)   | `file_0_view_0` | `Demand` |
| \(c_i\)   | `file_0_view_0` | `Initial Inventory` |
| Index \(i\) | `file_0_view_0` | `Product Name` |

*(No record values or counts are reproduced; only the exact table and column names are referenced.)*