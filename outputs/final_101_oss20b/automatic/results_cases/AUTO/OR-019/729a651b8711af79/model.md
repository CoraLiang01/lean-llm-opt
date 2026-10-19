**Abstract Model**

**Index Set**  
\(I=\{i \mid \text{product } i \text{ is classified as } 27\text{in}\}\)

**Parameters**  
\[
\begin{aligned}
w_i &\;=\; \text{Revenue per unit of product } i \quad (\text{from column } \texttt{Revenue})\\
d_i &\;=\; \text{Demand for product } i \quad (\text{from column } \texttt{Demand})\\
c_i &\;=\; \text{Initial inventory of product } i \quad (\text{from column } \texttt{Initial Inventory})
\end{aligned}
\]

**Decision Variables**  
\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\quad \forall i\in I
\]

**Objective Function**  
\[
\max \;\sum_{i\in I} w_i\,x_i
\]

**Constraints**

1. **Demand Constraints**  
\[
x_i \;\le\; d_i \quad \forall i\in I
\]

2. **Inventory Constraints**  
\[
x_i \;\le\; c_i \quad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(w_i\)   | `file_0_view_0` | `Revenue` |
| \(d_i\)   | `file_0_view_0` | `Demand` |
| \(c_i\)   | `file_0_view_0` | `Initial Inventory` |
| Product Identifier | `file_0_view_0` | `Product Name` (used to determine membership in \(I\) via the substring `27in`) |

*The set \(I\) is obtained by selecting rows where the `Product Name` column contains the substring `27in`.*