**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} p_i\,x_i \\[4pt]
\text{subject to}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{+}\qquad \forall i\in I
\end{aligned}
\]

**Index set**

- \(I\) – set of bread types (one element per product).

**Parameters**

- \(p_i\) – profit per unit of bread type \(i\) (from *products.csv* column **Value**).  
- \(w_i\) – weight (storage space) per unit of bread type \(i\) (from *products.csv* column **Weight**).  
- \(C\) – total storage capacity (from *capacity.csv* column **Capacity**).

**Decision variables**

- \(x_i\) – number of units of bread type \(i\) to order (integer, non‑negative).

---

### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(p_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(I\) | `file_1_view_0` | `ProductName` |

All mappings use the exact table identifiers and column names provided in the observation. No additional data or constraints are introduced beyond those specified.