Let x_{ij} be binary variables indicating assignment of manager i to project j. The cost matrix C is:

C = 
| 2972  2727  2795  2922  1302  2489  1533 |
| 1094  2158  2990  1844  2887  2021  2288 |
| 2133  1675  2422  2639  1033  2261  1695 |
| 1951  2309  2070  2802  2328  1313  2434 |
| 1269  2153  1296  2685  2627  1610  1641 |
| 1220  1192  2907  2622  2595  1261  2384 |
| 1286  1659  1179  1348  1420  2862  1959 |

The mathematical model is:

Minimize
\[
\sum_{i=1}^{7} \sum_{j=1}^{7} c_{ij} x_{ij}
\]

Subject to
\[
\sum_{j=1}^{7} x_{ij} = 1 \quad \forall i=1,\ldots,7
\]
\[
\sum_{i=1}^{7} x_{ij} = 1 \quad \forall j=1,\ldots,7
\]
\[
x_{ij} \in \{0,1\} \quad \forall i,j
\]

where c_{ij} is as given in the matrix above. This model assigns each manager to exactly one project and each project to exactly one manager, minimizing the total cost.