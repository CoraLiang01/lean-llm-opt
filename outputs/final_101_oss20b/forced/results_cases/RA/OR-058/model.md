**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}=d_j \qquad &&\forall j\in J \\[4pt]
&x_{ij}\;\le\; d_j\,y_i \qquad &&\forall i\in I,\; j\in J \\[4pt]
&y_i\in\{0,1\} \qquad &&\forall i\in I \\[4pt]
&x_{ij}\ge 0 \qquad &&\forall i\in I,\; j\in J
\end{aligned}
\]

**Index Sets**

| Symbol | Description | Source |
|--------|-------------|--------|
| \(I\) | Suppliers | `file_1_view_0` column `Unnamed: 0` |
| \(J\) | Customers | `file_0_view_0` column `customer` |

**Parameters**

| Symbol | Meaning | Data Mapping |
|--------|---------|--------------|
| \(d_j\) | Demand of customer \(j\) | `file_0_view_0` column `demand` (indexed by `customer`) |
| \(f_i\) | Fixed opening cost of supplier \(i\) | `file_1_view_0` column `fixed_costs` (indexed by `Unnamed: 0`) |
| \(c_{ij}\) | Transportation cost per unit from supplier \(i\) to customer \(j\) | `file_2_view_0` columns `C1`–`C6` (indexed by `Unnamed: 0` for suppliers and `customer` for customers) |

**Variables**

| Symbol | Domain | Interpretation |
|--------|--------|----------------|
| \(y_i\) | \(\{0,1\}\) | 1 if supplier \(i\) is opened, 0 otherwise |
| \(x_{ij}\) | \(\mathbb{R}_{\ge 0}\) | Quantity shipped from supplier \(i\) to customer \(j\) |

The linking constraint \(x_{ij}\le d_j\,y_i\) ensures that no shipment is sent from a supplier that is not opened. The demand‑satisfaction constraints guarantee that each customer’s demand is met exactly. The objective minimizes the sum of fixed opening costs and variable transportation costs.