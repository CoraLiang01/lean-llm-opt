**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) – set of shelves (index given by *ShelfID*).  
* \(J=\{1,\dots ,20\}\) – set of products (index given by *ProductName*).  

Parameters  

* \(\text{Cap}_i\) – capacity of shelf \(i\) (units of weight).  
* \(\text{Val}_j\) – value of one unit of product \(j\).  
* \(\text{Wgt}_j\) – weight of one unit of product \(j\).  

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\quad\forall i\in I,\;j\in J
\]

---

### Objective  

\[
\max \;\; \sum_{i\in I}\sum_{j\in J} \text{Val}_j\,x_{ij}
\]

### Constraints  

1. **Shelf capacity** – weight on each shelf cannot exceed its capacity  

\[
\sum_{j\in J} \text{Wgt}_j\,x_{ij}\;\le\;\text{Cap}_i
\qquad\forall i\in I
\]

2. **Non‑negativity & integrality** – already encoded in the variable definition.  

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Index Column | Notes |
|-----------|--------------|---------------|--------------|-------|
| \(\text{Cap}_i\) | `file_0_view_0` | `Capacity` | `ShelfID` | Capacity in units of weight |
| \(\text{Val}_j\) | `file_1_view_0` | `Value` | `ProductName` | Monetary value per unit |
| \(\text{Wgt}_j\) | `file_1_view_0` | `Weight` | `ProductName` | Weight per unit |

All indices \(i\) and \(j\) are taken directly from the unique identifiers in the respective tables. No additional aggregation or transformation is required.