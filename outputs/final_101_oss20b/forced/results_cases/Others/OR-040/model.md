## Mathematical Model

**Index set**  
\(I=\{i \mid i \text{ is a ProductName in *products.csv*}\}\)

**Parameters**  
\(b_i\) – benefit coefficient for area \(i\) (from *products.csv* column **Value**)  
\(C\) – overall development capacity (from *capacity.csv* column **Capacity**)

**Decision variables**  
\(x_i \in \mathbb{Z}_{+}\) – daily scale of development in area \(i\)

**Objective**  
\[
\max \;\sum_{i\in I} b_i\,x_i
\]

**Constraints**  
\[
\sum_{i\in I} x_i \;\le\; C
\]

---

## Data Mapping

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(b_i\) | `file_1_view_0` | `Value` | One entry per `ProductName` |
| \(C\) | `file_0_view_0` | `Capacity` | Single scalar value |

All other data (e.g., product names) are used only to define the index set \(I\). No additional constraints or variables are introduced.