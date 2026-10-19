##### Abstract Model

**Sets**  
- \(I\): set of development areas.

**Parameters**  
- \(b_i\): benefit coefficient for area \(i\) (from *products.csv*).  
- \(C\): overall development capacity (from *capacity.csv*).

**Variables**  
- \(x_i \in \mathbb{Z},\; x_i \ge 0,\; \forall i \in I\).

**Objective**  
\[
\max \sum_{i \in I} b_i \, x_i
\]

**Constraints**  

1. Capacity constraint  
\[
\sum_{i \in I} x_i \le C
\]

##### Data Mapping

| Source | Table ID | Column(s) Used |
|--------|----------|----------------|
| *capacity.csv* | `file_0_view_0` | `Capacity` |
| *products.csv* | `file_1_view_0` | `ProductName`, `Value` |