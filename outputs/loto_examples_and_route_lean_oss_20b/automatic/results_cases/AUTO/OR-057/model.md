## Mathematical Model

**Indices**  
\(I=\{1,\dots ,15\}\) – platforms (PlatformID)  
\(J=\{1,\dots ,15\}\) – products (ProductName)

**Parameters**  
\(C_i\) – memory capacity of platform \(i\) (Capacity)  
\(V_j\) – value of product \(j\) (Value)  
\(W_j\) – memory weight of product \(j\) (Weight)

**Decision Variables**  
\(x_{ij}\in \mathbb{Z}_{+}\) – number of units of product \(j\) listed on platform \(i\)

**Objective**  
\[
\max \;\sum_{i\in I}\sum_{j\in J} V_j\,x_{ij}
\]

**Constraints**  
\[
\sum_{j\in J} W_j\,x_{ij}\;\le\; C_i \qquad \forall\, i\in I
\]

\[
x_{ij}\;\ge\;0 \quad\text{and integer}\qquad \forall\, i\in I,\; j\in J
\]

---

## Data Mapping Rules

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(I\) | `file_0_view_0` | `PlatformID` | Set of all PlatformID values |
| \(J\) | `file_1_view_0` | `ProductName` | Set of all ProductName values |
| \(C_i\) | `file_0_view_0` | `Capacity` | For each row where `PlatformID = i`, assign `Capacity` to \(C_i\) |
| \(V_j\) | `file_1_view_0` | `Value` | For each row where `ProductName = j`, assign `Value` to \(V_j\) |
| \(W_j\) | `file_1_view_0` | `Weight` | For each row where `ProductName = j`, assign `Weight` to \(W_j\) |

All variables \(x_{ij}\) are defined for every combination of \(i\in I\) and \(j\in J\). No additional constraints or decision variables are introduced beyond those specified.