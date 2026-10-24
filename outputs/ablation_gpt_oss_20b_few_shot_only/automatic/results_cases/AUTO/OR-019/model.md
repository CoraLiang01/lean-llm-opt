**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad & \sum_{i\in I} \text{rev}[i]\;x[i] \\[4pt]
\text{subject to}\quad
& x[i] \;\le\; \text{dem}[i] && \forall i\in I \\[2pt]
& x[i] \;\le\; \text{inv}[i] && \forall i\in I \\[2pt]
& x[i] \;\in\; \mathbb{Z}_{+} && \forall i\in I
\end{aligned}
\]

**Index Set**

- \(I\): set of all products whose *Product Name* contains the substring **“27in”**.

**Parameters**

| Symbol | Meaning | Data Source |
|--------|---------|-------------|
| \(\text{rev}[i]\) | Revenue per unit of product \(i\) | `Revenue` column of *SalesDataAnalysis.csv* |
| \(\text{dem}[i]\) | Demand for product \(i\) | `Demand` column of *SalesDataAnalysis.csv* |
| \(\text{inv}[i]\) | Initial inventory of product \(i\) | `Initial Inventory` column of *SalesDataAnalysis.csv* |

**Variables**

- \(x[i]\) – integer number of units of product \(i\) to sell (non‑negative).

---

### Data Mapping Rules

| Parameter | File Path | Column Name |
|-----------|-----------|-------------|
| \(\text{rev}[i]\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv` | `Revenue` |
| \(\text{dem}[i]\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv` | `Demand` |
| \(\text{inv}[i]\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv` | `Initial Inventory` |

*Only rows where the `Product Name` contains “27in” are included in the index set \(I\). No other data are used.*