**Mathematical Model**

Let  

- \(I\) be the set of storage areas.  
- \(J\) be the set of air‑conditioner product types.  

| Symbol | Meaning | Domain |
|--------|---------|--------|
| \(i\) | Storage area index | \(i \in I\) |
| \(j\) | Product type index | \(j \in J\) |
| \(C_i\) | Capacity of storage area \(i\) | \(\mathbb{R}_{+}\) |
| \(V_j\) | Value of one unit of product \(j\) | \(\mathbb{R}_{+}\) |
| \(W_j\) | Weight (size) of one unit of product \(j\) | \(\mathbb{R}_{+}\) |
| \(x_{ij}\) | Integer number of units of product \(j\) placed in area \(i\) | \(\mathbb{Z}_{+}\) |

**Objective**

\[
\max \;\; \sum_{i\in I}\;\sum_{j\in J}\; V_j \, x_{ij}
\]

**Constraints**

1. **Capacity limits (per storage area)**  

\[
\sum_{j\in J} W_j \, x_{ij} \;\le\; C_i , \qquad \forall i\in I
\]

2. **Non‑negativity and integrality**  

\[
x_{ij}\;\in\;\mathbb{Z}_{+}, \qquad \forall i\in I,\; j\in J
\]

---

### Data Mapping

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C_i\) | `file_0_view_0` | `Capacity` |
| \(V_j\) | `file_1_view_0` | `Value` |
| \(W_j\) | `file_1_view_0` | `Weight` |

*No other data are required for the model.*