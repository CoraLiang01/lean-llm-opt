## Mathematical Model

**Index sets**  
\(I=\{\,i\mid i=\text{ShelfID in capacity.csv}\,\}\) (10 shelves)  
\(J=\{\,j\mid j=\text{ProductName in products.csv}\,\}\) (20 products)

**Parameters**  
\(\text{Cap}_i\) – capacity of shelf \(i\) (kg) (from *capacity.csv*)  
\(\text{Val}_j\) – value per unit of product \(j\) (from *products.csv*)  
\(\text{Wgt}_j\) – weight per unit of product \(j\) (from *products.csv*)

**Decision variables**  
\(x_{ij}\in\mathbb{Z}_{\ge 0}\) – number of units of product \(j\) placed on shelf \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Val}_j\,x_{ij}
\]

**Constraints**  
For every shelf \(i\in I\):
\[
\sum_{j\in J}\text{Wgt}_j\,x_{ij}\;\le\;\text{Cap}_i
\]

All other variables are non‑negative integers.

---

## Data Mapping Rules

| Parameter | Source Table | Source Column | Mapping Expression |
|-----------|--------------|---------------|--------------------|
| \(\text{Cap}_i\) | `file_0_view_0` | `Capacity` | \(\text{Cap}_i = \text{Capacity}\) where `ShelfID = i` |
| \(\text{Val}_j\) | `file_1_view_0` | `Value` | \(\text{Val}_j = \text{Value}\) where `ProductName = j` |
| \(\text{Wgt}_j\) | `file_1_view_0` | `Weight` | \(\text{Wgt}_j = \text{Weight}\) where `ProductName = j` |

- Index set \(I\) consists of all `ShelfID` values in `file_0_view_0`.  
- Index set \(J\) consists of all `ProductName` values in `file_1_view_0`.  
- No filtering or additional constraints are applied beyond those listed.  
- All parameters are real numbers; decision variables are integers.