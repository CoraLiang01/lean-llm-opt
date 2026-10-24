**Mathematical Model**

Let  

* \(I=\{MA,MB,MC,MD,ME,MF\}\) (Managers)  
* \(J=\{P1,P2,P3,P4,P5,P6\}\) (Projects)

Parameters  

* \(c_{ij}\) – assignment cost of manager \(i\) to project \(j\) (obtained from the CSV).

Decision variables  

* \(x_{ij}\in\{0,1\}\) – equals 1 if manager \(i\) is assigned to project \(j\), 0 otherwise.

\[
\begin{aligned}
\min\;&\sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;&\sum_{j\in J} x_{ij}=1 &&\forall i\in I \quad\text{(each manager gets one project)}\\
&\sum_{i\in I} x_{ij}=1 &&\forall j\in J \quad\text{(each project gets one manager)}\\
&x_{ij}\in\{0,1\} &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Table ID | Source Column | Target Index / Parameter |
|----------|---------------|--------------------------|
| `file_0_view_0` | `Unnamed: 0` | Manager index \(i\in I\) |
| `file_0_view_0` | `P1` | \(c_{i,P1}\) |
| `file_0_view_0` | `P2` | \(c_{i,P2}\) |
| `file_0_view_0` | `P3` | \(c_{i,P3}\) |
| `file_0_view_0` | `P4` | \(c_{i,P4}\) |
| `file_0_view_0` | `P5` | \(c_{i,P5}\) |
| `file_0_view_0` | `P6` | \(c_{i,P6}\) |

*Each row of the table supplies one manager \(i\); each column `Pk` supplies the corresponding cost \(c_{i,Pk}\). No additional aggregation or transformation is applied.*