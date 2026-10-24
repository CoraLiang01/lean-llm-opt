**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I}\sum_{j\in J} \text{Value}_j \; x_{ij} \\[4pt]
\text{subject to}\quad
& \sum_{j\in J} \text{Weight}_j \; x_{ij} \;\le\; \text{Capacity}_i,
   && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0},
   && \forall i\in I,\; j\in J
\end{aligned}
\]

**Index Sets**

- \(I = \{\, i \mid i \text{ is a ShelfID in capacity.csv}\,\}\)
- \(J = \{\, j \mid j \text{ is a ProductName in products.csv}\,\}\)

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(\text{Capacity}_i\) | Shelf capacity | `file_0_view_0`, column `Capacity`, key `ShelfID` |
| \(\text{Value}_j\) | Product value | `file_1_view_0`, column `Value`, key `ProductName` |
| \(\text{Weight}_j\) | Product weight | `file_1_view_0`, column `Weight`, key `ProductName` |

**Data Mapping Rules**

- `file_0_view_0` provides the set \(I\) via column `ShelfID` and the parameter \(\text{Capacity}_i\) via column `Capacity`.
- `file_1_view_0` provides the set \(J\) via column `ProductName` and the parameters \(\text{Value}_j\) and \(\text{Weight}_j\) via columns `Value` and `Weight`, respectively.

No other parameters or constraints are required.