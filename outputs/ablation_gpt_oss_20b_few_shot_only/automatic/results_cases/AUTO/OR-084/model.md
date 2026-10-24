## Mathematical Model

**Indices**  
\(i \in \{1,2,3\}\) – CPUs (1 = 1.33 GHz, 2 = 2 GHz, 3 = 2.66 GHz)  
\(j \in \{1,\dots ,40\}\) – tasks  

**Parameters**  
\(\text{BI}_j\) – basic instructions for task \(j\) (billions of instructions)  
\(f_i\) – frequency of CPU \(i\) (GHz)  
\(p_{ij} = \dfrac{\text{BI}_j}{f_i}\) – processing time of task \(j\) on CPU \(i\) (seconds)

**Decision Variables**  
\(x_{ij}\in\{0,1\}\) – 1 if task \(j\) is assigned to CPU \(i\), 0 otherwise  
\(T\ge 0\) – makespan (completion time of the last task)

**Objective**  
\[
\min \; T
\]

**Constraints**

1. **Task assignment**  
\[
\sum_{i=1}^{3} x_{ij}=1 \qquad \forall j=1,\dots ,40
\]

2. **CPU load (makespan bound)**  
\[
\sum_{j=1}^{40} p_{ij}\,x_{ij}\;\le\; T \qquad \forall i=1,2,3
\]

3. **Variable domains**  
\[
x_{ij}\in\{0,1\}\quad \forall i,j;\qquad T\ge 0
\]

---

## Data Mapping

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(\text{BI}_j\) | `Test_Dataset/Large-scale-or/Others_example/Others6/18.csv` | Row labeled **BI**, column **j** (j = 1…40) |
| \(f_1\) | Model definition | 1.33 |
| \(f_2\) | Model definition | 2.00 |
| \(f_3\) | Model definition | 2.66 |
| \(p_{ij}\) | Derived | \(\text{BI}_j / f_i\) |

No other data are required. The model uses only the basic‑instruction values from the CSV and the fixed CPU frequencies.