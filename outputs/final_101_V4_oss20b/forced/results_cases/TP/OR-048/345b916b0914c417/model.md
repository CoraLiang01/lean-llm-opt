**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I}\sum_{j\in J} V_j\,x_{ij} \\[4pt]
\text{subject to}\quad
& \sum_{j\in J} W_j\,x_{ij}\;\le\; C_i , &&\forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{+}, &&\forall i\in I,\; j\in J
\end{aligned}
\]

**Index Sets**

- \(I = \{\,i \mid \text{StorageID } i \text{ appears in capacity.csv}\,\}\)
- \(J = \{\,j \mid \text{ProductName } j \text{ appears in products.csv}\,\}\)

**Parameters**

- \(C_i\) – capacity of storage area \(i\) (from column **Capacity** in *capacity.csv*).
- \(V_j\) – value of product \(j\) (from column **Value** in *products.csv*).
- \(W_j\) – weight (size) of product \(j\) (from column **Weight** in *products.csv*).

**Decision Variables**

- \(x_{ij}\) – number of units of product \(j\) placed in storage area \(i\) (integer, \(\ge 0\)).

---

**Data Mapping Rules**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C_i\)   | `file_0_view_0` | `Capacity` |
| \(V_j\)   | `file_1_view_0` | `Value` |
| \(W_j\)   | `file_1_view_0` | `Weight` |

*Indices \(i\) and \(j\) are taken directly from the `StorageID` and `ProductName` columns of the respective tables.*