## Mathematical Model

**Sets**  
\(P=\{1,\dots ,15\}\) platforms (index \(i\))  
\(J=\{1,\dots ,15\}\) products (index \(j\))

**Parameters**  
\(\text{Capacity}[i]\) memory capacity of platform \(i\) (from *capacity.csv*)  
\(\text{Value}[j]\) value per unit of product \(j\) (from *products.csv*)  
\(\text{Weight}[j]\) memory required per unit of product \(j\) (from *products.csv*)

**Decision variables**  
\(x_{ij}\in\mathbb{Z}_{\ge 0}\) number of units of product \(j\) listed on platform \(i\)

**Objective**  
\[
\max \;\sum_{i\in P}\sum_{j\in J}\text{Value}[j]\;x_{ij}
\]

**Constraints**  
For every platform \(i\in P\):
\[
\sum_{j\in J}\text{Weight}[j]\;x_{ij}\;\le\;\text{Capacity}[i]
\]

No other constraints are required.

---

## Data Mapping Rules

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(\text{Capacity}[i]\) | `file_0_view_0` | `Capacity` | For each row where `PlatformID = i`, assign the numeric value of `Capacity`. |
| \(\text{Value}[j]\) | `file_1_view_0` | `Value` | For each row where `ProductName = j`, assign the numeric value of `Value`. |
| \(\text{Weight}[j]\) | `file_1_view_0` | `Weight` | For each row where `ProductName = j`, assign the numeric value of `Weight`. |

- Index sets \(P\) and \(J\) are defined by the distinct `PlatformID` and `ProductName` values present in the respective tables.
- All decision variables \(x_{ij}\) are integer and non‑negative.
- The model uses only the data and constraints explicitly stated in the query.