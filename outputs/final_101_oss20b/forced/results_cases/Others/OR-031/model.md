## Mathematical Model

**Index set**  
\(I=\{i\mid\text{product }i\text{ in } \text{DairyGoodsSalesDataset.csv}\}\)

**Parameters**  
\[
\begin{aligned}
r_i &\;=\;\text{Revenue per unit of product }i \quad (\$)\\
d_i &\;=\;\text{Demand for product }i \quad (\text{units})\\
\text{inv}_i &\;=\;\text{Initial inventory of product }i \quad (\text{units})
\end{aligned}
\]

**Decision variables**  
\[
x_i \;\ge\;0 \qquad\text{(units of product }i\text{ to fulfill)}
\]

**Objective**  
\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**  

1. **Demand limit**  
\[
x_i \;\le\; d_i \qquad \forall i\in I
\]

2. **Inventory limit**  
\[
x_i \;\le\; \text{inv}_i \qquad \forall i\in I
\]

3. **Non‑negativity**  
\[
x_i \;\ge\; 0 \qquad \forall i\in I
\]

---

## Data Mapping

| Parameter | Source Table | Column | Mapping Rule |
|-----------|--------------|--------|--------------|
| \(r_i\) | `file_0_view_0` | `Revenue` | For each product row \(i\), \(r_i =\) value in `Revenue`. |
| \(d_i\) | `file_0_view_0` | `Demand` | For each product row \(i\), \(d_i =\) value in `Demand`. |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` | For each product row \(i\), \(\text{inv}_i =\) value in `Initial Inventory`. |

*Index set \(I\) is derived from all rows returned in `file_0_view_0` (40 products). No additional filtering is applied.*