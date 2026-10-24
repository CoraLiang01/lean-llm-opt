## Mathematical Model

**Sets**  
\(I=\{i\mid i\text{ is a ShelfID in }file\_0\_view\_0\}\)  \(J=\{j\mid j\text{ is a ProductName in }file\_1\_view\_0\}\)

**Parameters**  
\(C_i\) – capacity of shelf \(i\)  \(w_j\) – weight of product \(j\)  \(v_j\) – value of product \(j\)

**Decision Variables**  
\(x_{ij}\in\mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed on shelf \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

**Constraints**  
\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; C_i \qquad \forall i\in I
\]

\[
x_{ij}\;\ge\;0,\; x_{ij}\in\mathbb{Z}\qquad \forall i\in I,\; j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Column | Mapping Rule |
|-----------|--------------|--------|--------------|
| \(C_i\) | `file_0_view_0` | `Capacity` | For each row, set \(C_{\text{ShelfID}} = \text{Capacity}\) |
| \(w_j\) | `file_1_view_0` | `Weight` | For each row, set \(w_{\text{ProductName}} = \text{Weight}\) |
| \(v_j\) | `file_1_view_0` | `Value` | For each row, set \(v_{\text{ProductName}} = \text{Value}\) |

Indices \(I\) and \(J\) are taken from the full set of rows in the respective tables; no additional filtering is applied.