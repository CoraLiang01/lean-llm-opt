**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{j\in \mathcal{J}} \; \text{profit}_{j}\;x_{j} \\[4pt]
\text{subject to}\quad
& \sum_{j\in \mathcal{J}} \; \text{time}_{i,j}\;x_{j}\;\le\;\text{capacity}_{i},
&& \forall\, i\in \mathcal{I}_{\text{cap}} \\[4pt]
& x_{j}\;\ge\;0, && \forall\, j\in \mathcal{J}
\end{aligned}
\]

**Index Sets**

| Symbol | Description | Elements |
|--------|-------------|----------|
| \(\mathcal{I}\) | Devices | \(\{A,B,C,D,E,F,G,H,I,J\}\) |
| \(\mathcal{J}\) | Products | \(\{P1,P2,\dots ,P111\}\) |
| \(\mathcal{I}_{\text{cap}}\) | Devices with capacity data | \(\{A,B,I,J\}\) |

**Parameters**

| Symbol | Source | Column(s) | Notes |
|--------|--------|-----------|-------|
| \(\text{profit}_{j}\) | `file_2_view_0` | `Unit_Profit` | Key: `Product` |
| \(\text{time}_{i,j}\) | `file_0_view_0` | matrix of numeric values | Row key: `Device`, Column key: `Product` |
| \(\text{capacity}_{i}\) | `file_1_view_0` | `Monthly_Capacity` | Key: `Device` (only for \(i\in\mathcal{I}_{\text{cap}}\)) |

**Decision Variables**

| Symbol | Domain | Meaning |
|--------|--------|---------|
| \(x_{j}\) | \(\mathbb{R}_{+}\) | Monthly production quantity of product \(j\) |

---

### Data Mapping

| Parameter | Table ID | Key Column | Value Column |
|-----------|----------|------------|--------------|
| \(\text{profit}_{j}\) | `file_2_view_0` | `Product` | `Unit_Profit` |
| \(\text{time}_{i,j}\) | `file_0_view_0` | `Device` (row) | `Product` (column) |
| \(\text{capacity}_{i}\) | `file_1_view_0` | `Device` | `Monthly_Capacity` |

*Only devices listed in `file_1_view_0` (A, B, I, J) receive a capacity value; constraints are applied only to those devices.*