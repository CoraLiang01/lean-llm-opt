## Mathematical Model

**Sets**  
\(I\) – set of cabinet identifiers (taken from *capacity.csv*)  
\(J\) – set of product names (taken from *products.csv*)

**Parameters**  
\(\text{Cap}[i]\) – capacity of cabinet \(i\) (kg)  
\(\text{Val}[j]\) – value of one unit of product \(j\) (\$)  
\(\text{Wgt}[j]\) – weight of one unit of product \(j\) (kg)

**Decision variables**  
\(x_{ij}\) – number of units of product \(j\) placed in cabinet \(i\)  
\(x_{ij}\in\mathbb{Z}_{\ge 0}\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Val}[j]\;x_{ij}
\]

**Constraints**  

*Weight capacity of each cabinet*  
\[
\sum_{j\in J}\text{Wgt}[j]\;x_{ij}\;\le\;\text{Cap}[i]\qquad \forall i\in I
\]

*Non‑negativity and integrality*  
\[
x_{ij}\in\mathbb{Z}_{\ge 0}\qquad \forall i\in I,\;j\in J
\]

---

## Data Mapping

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(\text{Cap}[i]\) | `file_0_view_0` | `Capacity` | For each record, map `CabinetID` → `i`, `Capacity` → \(\text{Cap}[i]\). |
| \(\text{Val}[j]\) | `file_1_view_0` | `Value` | For each record, map `ProductName` → `j`, `Value` → \(\text{Val}[j]\). |
| \(\text{Wgt}[j]\) | `file_1_view_0` | `Weight` | For each record, map `ProductName` → `j`, `Weight` → \(\text{Wgt}[j]\). |

*Index sets*  
\(I = \{\,\text{CabinetID}\text{ values from } \text{file\_0\_view\_0}\,\}\)  
\(J = \{\,\text{ProductName}\text{ values from } \text{file\_1\_view\_0}\,\}\)

No additional constraints or transformations are required beyond those listed in the model.