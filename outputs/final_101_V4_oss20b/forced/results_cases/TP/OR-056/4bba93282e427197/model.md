**Mathematical Model**

\[
\begin{aligned}
\max\;&\sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{j\in J} w_j\,x_{ij}\;\le\;c_i , &&\forall i\in I \\[4pt]
&x_{ij}\;\in\;\mathbb{Z}_{+}, &&\forall i\in I,\;j\in J
\end{aligned}
\]

**Index Sets**

| Symbol | Description | Source |
|--------|-------------|--------|
| \(I\) | Display area identifiers | `file_0_view_0` (column `DisplayID`) |
| \(J\) | Vessel type identifiers | `file_1_view_0` (column `ProductName`) |

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(c_i\) | Capacity of display area \(i\) | `file_0_view_0` (column `Capacity`) |
| \(v_j\) | Monetary value of vessel type \(j\) | `file_1_view_0` (column `Value`) |
| \(w_j\) | Physical weight/size of vessel type \(j\) | `file_1_view_0` (column `Weight`) |

**Decision Variables**

| Symbol | Meaning | Domain |
|--------|---------|--------|
| \(x_{ij}\) | Number of vessels of type \(j\) placed in display area \(i\) | \(\mathbb{Z}_{+}\) |

---

**Data Mapping Rules**

| Table ID | Column | Parameter / Variable |
|----------|--------|----------------------|
| `file_0_view_0` | `DisplayID` | Index set \(I\) |
| `file_0_view_0` | `Capacity` | \(c_i\) |
| `file_1_view_0` | `ProductName` | Index set \(J\) |
| `file_1_view_0` | `Value` | \(v_j\) |
| `file_1_view_0` | `Weight` | \(w_j\) |

All mappings use the exact identifiers from the source tables; no aggregation or renaming is performed. The model is ready for numerical instantiation with the provided CSV data.