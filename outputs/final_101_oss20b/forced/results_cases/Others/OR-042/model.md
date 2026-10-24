## Mathematical Model

**Index set**  
\(I=\{i\mid\text{product }i\text{ appears in } \text{products.csv}\}\)

**Parameters**  
\(\text{Value}_i\) – benefit coefficient of product \(i\) (from *products.csv*)  
\(\text{Weight}_i\) – weight of one unit of product \(i\) (from *products.csv*)  
\(\text{Capacity}\) – total inventory capacity (from *capacity.csv*)

**Decision variables**  
\(x_i\in\mathbb{Z}_{\ge 0}\) – number of units of product \(i\) ordered daily

**Objective**  
\[
\max \;\sum_{i\in I}\text{Value}_i\,x_i
\]

**Constraint**  
\[
\sum_{i\in I}\text{Weight}_i\,x_i \;\le\; \text{Capacity}
\]

---

## Data Mapping

| Table ID | Source File | Columns Used | Mapping |
|----------|-------------|--------------|---------|
| file_0_view_0 | capacity.csv | Capacity | \(\text{Capacity}\) |
| file_1_view_0 | products.csv | ProductName, Value, Weight | \(\text{Value}_{\text{ProductName}}\), \(\text{Weight}_{\text{ProductName}}\) |

*Index set \(I\) is the set of all `ProductName` values returned from `products.csv`.*