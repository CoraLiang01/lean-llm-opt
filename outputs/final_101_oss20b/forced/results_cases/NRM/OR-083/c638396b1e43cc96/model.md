**Abstract Model**

**Index Sets**  
\(I\) – set of workers, indexed by the values in column **Task Time Required** (1,…,12).  
\(J\) – set of tasks, indexed by the columns **A**,…,**J** (10 tasks).

**Parameters**  
\(t_{ij}\) – time required for worker \(i\) to complete task \(j\).  
\(M\) – a sufficiently large constant (e.g., \(M=10^6\)).

**Decision Variables**  
\(x_{ij}\in\{0,1\}\) – equals 1 if worker \(i\) is assigned to task \(j\), 0 otherwise.  

**Objective Function**  
\[
\min \;\sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

**Constraints**

1. **Task Assignment** – each task is assigned to exactly one worker  
\[
\sum_{i\in I} x_{ij}=1,\qquad \forall j\in J
\]

2. **Worker Capacity** – each worker is assigned to at most one task  
\[
\sum_{j\in J} x_{ij}\le 1,\qquad \forall i\in I
\]

3. **Number of Workers Used** – exactly ten workers are employed (implied by 1 and 2, but stated explicitly)  
\[
\sum_{i\in I}\sum_{j\in J} x_{ij}=10
\]

---

**Data Mapping**

| Table ID | Column Name | Description |
|----------|-------------|-------------|
| `file_0_view_0` | `Task Time Required` | Worker identifier (1–12) |
| `file_0_view_0` | `A` | Time for worker to complete Task A |
| `file_0_view_0` | `B` | Time for worker to complete Task B |
| `file_0_view_0` | `C` | Time for worker to complete Task C |
| `file_0_view_0` | `D` | Time for worker to complete Task D |
| `file_0_view_0` | `E` | Time for worker to complete Task E |
| `file_0_view_0` | `F` | Time for worker to complete Task F |
| `file_0_view_0` | `G` | Time for worker to complete Task G |
| `file_0_view_0` | `H` | Time for worker to complete Task H |
| `file_0_view_0` | `I` | Time for worker to complete Task I |
| `file_0_view_0` | `J` | Time for worker to complete Task J |

*All parameter values \(t_{ij}\) are taken directly from the corresponding cells in the CSV file `15.csv`.*