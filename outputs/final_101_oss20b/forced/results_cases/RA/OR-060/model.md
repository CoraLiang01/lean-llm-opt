**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{i\in I} x_{ij}\;\ge\; d_j \qquad &&\forall j\in J \quad\text{(demand)}\\[4pt]
&x_{ij}\;\le\; d_j\,y_i \qquad &&\forall i\in I,\; j\in J \quad\text{(link to opening)}\\[4pt]
&x_{ij}\;\ge\;0 \qquad &&\forall i\in I,\; j\in J\\
&y_i\in\{0,1\} \qquad &&\forall i\in I
\end{aligned}
\]

---

### Index Sets
| Symbol | Description | Source |
|--------|-------------|--------|
| \(I\) | Set of suppliers | `file_1_view_0` column **Unnamed: 0** |
| \(J\) | Set of customers | `file_0_view_0` column **customer** |

---

### Parameters
| Symbol | Meaning | Data Mapping |
|--------|---------|--------------|
| \(d_j\) | Demand of customer \(j\) | `file_0_view_0` column **demand** |
| \(f_i\) | Fixed opening cost of supplier \(i\) | `file_1_view_0` column **fixed_costs** |
| \(c_{ij}\) | Transportation cost per unit from supplier \(i\) to customer \(j\) | `file_2_view_0` columns **C1**–**C12** (row identified by supplier \(i\)) |

---

### Variables
| Symbol | Domain | Interpretation |
|--------|--------|----------------|
| \(y_i\) | \(\{0,1\}\) | 1 if supplier \(i\) is opened, 0 otherwise |
| \(x_{ij}\) | \(\mathbb{R}_{+}\) | Quantity shipped from supplier \(i\) to customer \(j\) |

---

**Data Mapping Rules**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(d_j\) | `file_0_view_0` | `demand` |
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C12` (column name equals customer ID) |

These mappings directly bind each symbolic parameter to the corresponding CSV column, preserving the original identifiers and order. No numerical values are enumerated in the model; the solver will read the data via the specified table IDs and columns.