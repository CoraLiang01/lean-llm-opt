## Mathematical Model

**Sets**  
\(I=\{1,2,\dots ,14\}\) (display areas)  
\(J=\{\text{Speedboat},\text{Fishing Boat},\dots ,\text{Skiff}\}\) (product types)

**Parameters**  
\(\text{Capacity}[i]\) (available size in area \(i\)) – from *capacity.csv*  
\(\text{Value}[j]\) (value of one unit of product \(j\)) – from *products.csv*  
\(\text{Weight}[j]\) (size of one unit of product \(j\)) – from *products.csv*

**Decision Variables**  
\(x_{ij}\in \mathbb{Z}_{\ge 0}\) (number of vessels of type \(j\) placed in area \(i\))

**Objective**  
\[
\max \;\sum_{i\in I}\;\sum_{j\in J}\text{Value}[j]\;x_{ij}
\]

**Constraints**  
For every display area \(i\in I\):
\[
\sum_{j\in J}\text{Weight}[j]\;x_{ij}\;\le\;\text{Capacity}[i]
\]

All variables are non‑negative integers.

---

## Data Mapping

| Table ID | Source File | Columns Used | Mapping |
|----------|-------------|--------------|---------|
| `file_0_view_0` | `capacity.csv` | `DisplayID`, `Capacity` | \(\text{Capacity}[i] =\) value of `Capacity` where `DisplayID` = \(i\) |
| `file_1_view_0` | `products.csv` | `ProductName`, `Value`, `Weight` | \(\text{Value}[j] =\) value of `Value` where `ProductName` = \(j\); <br>\(\text{Weight}[j] =\) value of `Weight` where `ProductName` = \(j\) |

No additional constraints or limits are imposed beyond those derived from the data and the problem statement.