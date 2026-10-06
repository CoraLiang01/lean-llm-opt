##### Objective Function:

$\quad \min \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j=1}^{11} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,11\}$

$\sum_{i=1}^{11} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,11\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i,j$

##### Data Mapping

- Let $i$ index managers and $j$ index projects.
- Managers:  
  1: Manager 1  
  2: Manager 2  
  3: Manager 3  
  4: Manager 4  
  5: Manager 5  
  6: Manager 6  
  7: Manager 7  
  8: Manager 8  
  9: Manager 9  
  10: Manager 10  
  11: Manager 11  

- Projects:  
  1: Project 1  
  2: Project 2  
  3: Project 3  
  4: Project 4  
  5: Project 5  
  6: Project 6  
  7: Project 7  
  8: Project 8  
  9: Project 9  
  10: Project 10  
  11: Project 11  

- Cost matrix $c_{ij}$ (from CSV columns "Project k Cost" for manager $i$):

| $c_{ij}$           | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    | 11    |
|--------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| **Manager 1**      | 708   | 1948  | 2424  | 1068  | 729   | 199   | 1651  | 3174  | 3211  | 3167  | 1711  |
| **Manager 2**      | 1700  | 2670  | 1883  | 2534  | 1429  | 1173  | 777   | 248   | 1704  | 2603  | 1822  |
| **Manager 3**      | 160   | 755   | 3477  | 3122  | 2968  | 3023  | 1417  | 254   | 3175  | 2502  | 2595  |
| **Manager 4**      | 2213  | 1008  | 411   | 1199  | 418   | 1000  | 3148  | 1724  | 1984  | 1954  | 1805  |
| **Manager 5**      | 198   | 1721  | 1318  | 3194  | 3036  | 2938  | 3298  | 3332  | 1806  | 270   | 1893  |
| **Manager 6**      | 2375  | 1804  | 3174  | 1607  | 2168  | 1642  | 970   | 3433  | 1528  | 2696  | 2217  |
| **Manager 7**      | 2400  | 211   | 1172  | 425   | 1222  | 287   | 653   | 1466  | 479   | 2762  | 577   |
| **Manager 8**      | 272   | 2574  | 413   | 202   | 1220  | 2392  | 410   | 2250  | 2272  | 3260  | 2981  |
| **Manager 9**      | 2844  | 2775  | 357   | 2601  | 1627  | 125   | 1029  | 1354  | 2280  | 114   | 2161  |
| **Manager 10**     | 1222  | 296   | 3375  | 352   | 2167  | 2202  | 3139  | 2526  | 767   | 1873  | 1185  |
| **Manager 11**     | 2661  | 887   | 455   | 2552  | 1067  | 552   | 2991  | 1727  | 1639  | 3003  | 2161  |

- Decision variables:  
  $x_{ij} = \begin{cases} 1 & \text{if Manager } i \text{ is assigned to Project } j \\ 0 & \text{otherwise} \end{cases}$

##### Summary

Minimize total assignment cost by assigning each manager to exactly one project and each project to exactly one manager, using the cost matrix above.