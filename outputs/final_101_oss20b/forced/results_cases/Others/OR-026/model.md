## Mathematical Model

**Index set**  
\(I=\{i\mid\) product *i* is a Fashion item in the filtered table\(\}\)

**Parameters**  
\(\text{Rev}_i\) – revenue per unit of product *i* (RMB)  
\(\text{Dem}_i\) – deterministic demand of product *i* (units)  
\(\text{Inv}_i\) – initial inventory of product *i* (units)

**Decision variables**  
\(x_i\in\mathbb{Z}_{\ge 0}\) – number of units of product *i* to sell

**Objective**  
\[
\max \;\sum_{i\in I}\text{Rev}_i\,x_i
\]

**Constraints**  
\[
\begin{aligned}
x_i &\le \text{Dem}_i &&\forall i\in I \quad\text{(cannot exceed demand)}\\
x_i &\le \text{Inv}_i &&\forall i\in I \quad\text{(cannot exceed inventory)}\\
x_i &\ge 0 &&\forall i\in I
\end{aligned}
\]

---

## Data Mapping Rules

| Table ID | Column | Source | Filter |
|----------|--------|--------|--------|
| `file_0_view_0` | `Product Name` | `SupermarketSales.csv` | `contains "Fashion"` |
| `file_0_view_0` | `Revenue` | `SupermarketSales.csv` | – |
| `file_0_view_0` | `Demand` | `SupermarketSales.csv` | – |
| `file_0_view_0` | `Initial Inventory` | `SupermarketSales.csv` | – |

- **Index set \(I\)**: all rows returned after applying the filter on `Product Name`.
- **Parameter \(\text{Rev}_i\)**: value from column `Revenue` for row *i*.
- **Parameter \(\text{Dem}_i\)**: value from column `Demand` for row *i*.
- **Parameter \(\text{Inv}_i\)**: value from column `Initial Inventory` for row *i*.

No additional data or transformations are required. The model uses only the filtered rows and the specified columns.