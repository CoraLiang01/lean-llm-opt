**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \qquad\text{(demand)}\\[4pt]
&x_{ij}\;\le\; y_i\,d_j , &&\forall i\in I,\; j\in J \qquad\text{(linking)}\\[4pt]
&x_{ij}\;\ge\;0 , &&\forall i\in I,\; j\in J \\[4pt]
&y_i\in\{0,1\}, &&\forall i\in I
\end{aligned}
\]

**Index Sets**

| Symbol | Description | Source |
|--------|-------------|--------|
| \(I\)  | Warehouses | `file_1_view_0` – column `Unnamed: 0` |
| \(J\)  | Musicians / bands | `file_0_view_0` – column `customer` |

**Parameters**

| Symbol | Meaning | Data source (table_id, column) |
|--------|---------|--------------------------------|
| \(d_j\) | Demand of customer \(j\) | `file_0_view_0`, column `demand` |
| \(f_i\) | Fixed operating cost of warehouse \(i\) | `file_1_view_0`, column `fixed_costs` |
| \(c_{ij}\) | Unit transportation cost from warehouse \(i\) to customer \(j\) | `file_2_view_0`, columns `C1`, `C2`, `C3` (row identified by `Unnamed: 0` = \(i\)) |

**Decision Variables**

| Symbol | Domain | Interpretation |
|--------|--------|----------------|
| \(y_i\) | \(\{0,1\}\) | 1 if warehouse \(i\) is activated, 0 otherwise |
| \(x_{ij}\) | \(\mathbb{R}_+\) | Quantity shipped from warehouse \(i\) to customer \(j\) |

**Data‑Mapping Rules**

1. **Warehouses \(I\)**  
   - Extract distinct values from column `Unnamed: 0` of `file_1_view_0`.  
   - Each value becomes an index \(i\).

2. **Customers \(J\)**  
   - Extract distinct values from column `customer` of `file_0_view_0`.  
   - Each value becomes an index \(j\).

3. **Demand \(d_j\)**  
   - For each \(j\in J\), set \(d_j\) to the value in column `demand` of the row where `customer` = \(j\) in `file_0_view_0`.

4. **Fixed Cost \(f_i\)**  
   - For each \(i\in I\), set \(f_i\) to the value in column `fixed_costs` of the row where `Unnamed: 0` = \(i\) in `file_1_view_0`.

5. **Transportation Cost \(c_{ij}\)**  
   - For each pair \((i,j)\), locate the row in `file_2_view_0` where `Unnamed: 0` = \(i\).  
   - Set \(c_{ij}\) to the value in the column named after customer \(j\) (`C1`, `C2`, or `C3`).

All parameters are taken directly from the supplied CSV files; no additional aggregation or transformation is performed. The model uses the full set of entities returned by the observation.