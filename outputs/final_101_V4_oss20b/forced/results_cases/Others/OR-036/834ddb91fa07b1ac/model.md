## Mathematical Model

**Index set**  
\(I=\{\text{ProductName}\}\)  (all 25 vehicle types)

**Parameters**  
\(\text{Value}_i\) – benefit coefficient for product \(i\) (from *products.csv*)  
\(\text{Weight}_i\) – inventory weight of product \(i\) (from *products.csv*)  
\(\text{Capacity}\) – total inventory capacity (from *capacity.csv*)

**Decision variables**  
\(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) ordered daily

**Objective**  
\[
\max \;\sum_{i\in I} \text{Value}_i\,x_i
\]

**Constraints**  
\[
\sum_{i\in I} \text{Weight}_i\,x_i \;\le\; \text{Capacity}
\]

\[
x_i \;\ge\; 0,\;\; x_i \in \mathbb{Z}\quad \forall i\in I
\]

---

## Data Mapping

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(\text{Capacity}\) | `file_0_view_0` | `Capacity` | single scalar value |
| \(\text{Value}_i\) | `file_1_view_0` | `Value` | one value per `ProductName` |
| \(\text{Weight}_i\) | `file_1_view_0` | `Weight` | one value per `ProductName` |
| Index set \(I\) | `file_1_view_0` | `ProductName` | each distinct product name |

All parameters are taken directly from the corresponding CSV files; no additional data or transformations are introduced. The model uses only the variables and constraints specified in the query.