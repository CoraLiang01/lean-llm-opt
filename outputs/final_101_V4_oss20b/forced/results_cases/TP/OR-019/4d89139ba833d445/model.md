**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & Z = \sum_{i\in I} \text{Revenue}_i \; x_i \\[4pt]
\text{subject to}\quad
& 0 \;\le\; x_i \;\le\; \text{Demand}_i && \forall i\in I \\[2pt]
& 0 \;\le\; x_i \;\le\; \text{Inventory}_i && \forall i\in I \\[4pt]
& x_i \;\ge\; 0 && \forall i\in I
\end{aligned}
\]

**Index Sets, Parameters, and Variables**

| Symbol | Description | Type |
|--------|-------------|------|
| \(I\) | Set of 27in products | Set |
| \(\text{Revenue}_i\) | Revenue per unit of product \(i\) | Parameter |
| \(\text{Demand}_i\) | Demand quantity of product \(i\) | Parameter |
| \(\text{Inventory}_i\) | Initial inventory of product \(i\) | Parameter |
| \(x_i\) | Units of product \(i\) to fulfill | Decision variable, \(\ge 0\) |

**Data Mapping Rules**

| Table ID | Column in source | Parameter / Index |
|----------|------------------|-------------------|
| `file_0_view_0` | `Product Name` | Index \(i\) |
| `file_0_view_0` | `Revenue` | \(\text{Revenue}_i\) |
| `file_0_view_0` | `Demand` | \(\text{Demand}_i\) |
| `file_0_view_0` | `Initial Inventory` | \(\text{Inventory}_i\) |

*Source rows for the two products:*
- Row 1 → product “27in 4K Gaming Monitor”
- Row 2 → product “27in FHD Monitor”

All parameters are taken directly from the corresponding columns of the filtered view `file_0_view_0`. No additional data or constraints are introduced beyond those specified.