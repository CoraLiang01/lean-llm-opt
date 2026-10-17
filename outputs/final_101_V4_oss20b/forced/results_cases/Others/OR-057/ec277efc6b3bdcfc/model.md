## Mathematical Model

**Sets**  
\(P=\{1,\dots ,15\}\) – platforms (PlatformID)  
\(J=\{1,\dots ,15\}\) – products (ProductName)

**Parameters**  
\(\text{Cap}_i\) = Capacity of platform \(i\) (from *capacity.csv*)  
\(\text{Val}_j\) = Value of product \(j\) (from *products.csv*)  
\(\text{Wgt}_j\) = Memory weight of product \(j\) (from *products.csv*)

**Decision Variables**  
\(x_{ij}\in \mathbb{Z}_{\ge 0}\) = number of units of product \(j\) listed on platform \(i\)

**Objective**  
\[
\max \;\sum_{i\in P}\sum_{j\in J}\text{Val}_j\,x_{ij}
\]

**Constraints**  
\[
\sum_{j\in J}\text{Wgt}_j\,x_{ij}\;\le\;\text{Cap}_i
\qquad \forall\, i\in P
\]

No other constraints are required.

---

## Data Mapping Rules

| Parameter | Source Table | Source Columns | Mapping Expression |
|-----------|--------------|----------------|--------------------|
| \(\text{Cap}_i\) | `file_0_view_0` | `PlatformID`, `Capacity` | \(\text{Cap}_i = \text{Capacity}\) where `PlatformID = i` |
| \(\text{Val}_j\) | `file_1_view_0` | `ProductName`, `Value` | \(\text{Val}_j = \text{Value}\) where `ProductName = j` |
| \(\text{Wgt}_j\) | `file_1_view_0` | `ProductName`, `Weight` | \(\text{Wgt}_j = \text{Weight}\) where `ProductName = j` |

All indices \(i\) and \(j\) are taken from the full set of rows returned in the observation (15 platforms, 15 products). No filtering or additional constraints are applied beyond those explicitly stated in the objective and capacity constraints.