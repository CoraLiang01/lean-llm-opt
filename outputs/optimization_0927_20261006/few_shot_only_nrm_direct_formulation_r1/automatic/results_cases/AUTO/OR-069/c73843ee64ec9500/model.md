##### Decision Variables

Let $x_{ij} = \begin{cases} 1 & \text{if Manager } i \text{ is assigned to Project } j \\ 0 & \text{otherwise} \end{cases}$

for $i = 1, \ldots, 11$ (Managers), $j = 1, \ldots, 11$ (Projects).

##### Parameters

Let $c_{ij}$ be the cost of assigning Manager $i$ to Project $j$, as given below:

|            | Project 1 | Project 2 | Project 3 | Project 4 | Project 5 | Project 6 | Project 7 | Project 8 | Project 9 | Project 10 | Project 11 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|
| Manager 1  |   708     |   1948    |   2424    |   1068    |   729     |   199     |   1651    |   3174    |   3211    |   3167     |   1711     |
| Manager 2  |   1700    |   2670    |   1883    |   2534    |   1429    |   1173    |   777     |   248     |   1704    |   2603     |   1822     |
| Manager 3  |   160     |   755     |   3477    |   3122    |   2968    |   3023    |   1417    |   254     |   3175    |   2502     |   2595     |
| Manager 4  |   2213    |   1008    |   411     |   1199    |   418     |   1000    |   3148    |   1724    |   1984    |   1954     |   1805     |
| Manager 5  |   198     |   1721    |   1318    |   3194    |   3036    |   2938    |   3298    |   3332    |   1806    |   270      |   1893     |
| Manager 6  |   2375    |   1804    |   3174    |   1607    |   2168    |   1642    |   970     |   3433    |   1528    |   2696     |   2217     |
| Manager 7  |   2400    |   211     |   1172    |   425     |   1222    |   287     |   653     |   1466    |   479     |   2762     |   577      |
| Manager 8  |   272     |   2574    |   413     |   202     |   1220    |   2392    |   410     |   2250    |   2272    |   3260     |   2981     |
| Manager 9  |   2844    |   2775    |   357     |   2601    |   1627    |   125     |   1029    |   1354    |   2280    |   114      |   2161     |
| Manager 10 |   1222    |   296     |   3375    |   352     |   2167    |   2202    |   3139    |   2526    |   767     |   1873     |   1185     |
| Manager 11 |   2661    |   887     |   455     |   2552    |   1067    |   552     |   2991    |   1727    |   1639    |   3003     |   2161     |

##### Objective Function

$$
\min \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
$$

##### Constraints

1. **Each manager is assigned to exactly one project:**

$$
\sum_{j=1}^{11} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,11\}
$$

2. **Each project is assigned to exactly one manager:**

$$
\sum_{i=1}^{11} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,11\}
$$

3. **Variable domains:**

$$
x_{ij} \in \{0,1\} \quad \forall i \in \{1,2,\ldots,11\},\ j \in \{1,2,\ldots,11\}
$$

##### Retrieved Information

{
  "cost": {
    "Manager 1":   {"Project 1": 708,  "Project 2": 1948, "Project 3": 2424, "Project 4": 1068, "Project 5": 729,  "Project 6": 199,  "Project 7": 1651, "Project 8": 3174, "Project 9": 3211, "Project 10": 3167, "Project 11": 1711},
    "Manager 2":   {"Project 1": 1700, "Project 2": 2670, "Project 3": 1883, "Project 4": 2534, "Project 5": 1429, "Project 6": 1173, "Project 7": 777,  "Project 8": 248,  "Project 9": 1704, "Project 10": 2603, "Project 11": 1822},
    "Manager 3":   {"Project 1": 160,  "Project 2": 755,  "Project 3": 3477, "Project 4": 3122, "Project 5": 2968, "Project 6": 3023, "Project 7": 1417, "Project 8": 254,  "Project 9": 3175, "Project 10": 2502, "Project 11": 2595},
    "Manager 4":   {"Project 1": 2213, "Project 2": 1008, "Project 3": 411,  "Project 4": 1199, "Project 5": 418,  "Project 6": 1000, "Project 7": 3148, "Project 8": 1724, "Project 9": 1984, "Project 10": 1954, "Project 11": 1805},
    "Manager 5":   {"Project 1": 198,  "Project 2": 1721, "Project 3": 1318, "Project 4": 3194, "Project 5": 3036, "Project 6": 2938, "Project 7": 3298, "Project 8": 3332, "Project 9": 1806, "Project 10": 270,  "Project 11": 1893},
    "Manager 6":   {"Project 1": 2375, "Project 2": 1804, "Project 3": 3174, "Project 4": 1607, "Project 5": 2168, "Project 6": 1642, "Project 7": 970,  "Project 8": 3433, "Project 9": 1528, "Project 10": 2696, "Project 11": 2217},
    "Manager 7":   {"Project 1": 2400, "Project 2": 211,  "Project 3": 1172, "Project 4": 425,  "Project 5": 1222, "Project 6": 287,  "Project 7": 653,  "Project 8": 1466, "Project 9": 479,  "Project 10": 2762, "Project 11": 577},
    "Manager 8":   {"Project 1": 272,  "Project 2": 2574, "Project 3": 413,  "Project 4": 202,  "Project 5": 1220, "Project 6": 2392, "Project 7": 410,  "Project 8": 2250, "Project 9": 2272, "Project 10": 3260, "Project 11": 2981},
    "Manager 9":   {"Project 1": 2844, "Project 2": 2775, "Project 3": 357,  "Project 4": 2601, "Project 5": 1627, "Project 6": 125,  "Project 7": 1029, "Project 8": 1354, "Project 9": 2280, "Project 10": 114,  "Project 11": 2161},
    "Manager 10":  {"Project 1": 1222, "Project 2": 296,  "Project 3": 3375, "Project 4": 352,  "Project 5": 2167, "Project 6": 2202, "Project 7": 3139, "Project 8": 2526, "Project 9": 767,  "Project 10": 1873, "Project 11": 1185},
    "Manager 11":  {"Project 1": 2661, "Project 2": 887,  "Project 3": 455,  "Project 4": 2552, "Project 5": 1067, "Project 6": 552,  "Project 7": 2991, "Project 8": 1727, "Project 9": 1639, "Project 10": 3003, "Project 11": 2161}
  },
  "managers": [
    "Manager 1", "Manager 2", "Manager 3", "Manager 4", "Manager 5", "Manager 6",
    "Manager 7", "Manager 8", "Manager 9", "Manager 10", "Manager 11"
  ],
  "projects": [
    "Project 1", "Project 2", "Project 3", "Project 4", "Project 5", "Project 6",
    "Project 7", "Project 8", "Project 9", "Project 10", "Project 11"
  ]
}