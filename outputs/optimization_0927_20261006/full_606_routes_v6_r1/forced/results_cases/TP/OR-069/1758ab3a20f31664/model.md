##### Sets

Let $M$ be the set of managers:
$$
M = \{\text{Manager 1},\ \text{Manager 2},\ \text{Manager 3},\ \text{Manager 4},\ \text{Manager 5},\ \text{Manager 6},\ \text{Manager 7},\ \text{Manager 8},\ \text{Manager 9},\ \text{Manager 10},\ \text{Manager 11}\}
$$

Let $P$ be the set of projects:
$$
P = \{\text{Project 1},\ \text{Project 2},\ \text{Project 3},\ \text{Project 4},\ \text{Project 5},\ \text{Project 6},\ \text{Project 7},\ \text{Project 8},\ \text{Project 9},\ \text{Project 10},\ \text{Project 11}\}
$$

##### Parameters

Let $c_{ij}$ be the cost of assigning manager $i$ to project $j$, as given below (source order):

|            | Project 1 | Project 2 | Project 3 | Project 4 | Project 5 | Project 6 | Project 7 | Project 8 | Project 9 | Project 10 | Project 11 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|------------|------------|
| Manager 1  | 708       | 1948      | 2424      | 1068      | 729       | 199       | 1651      | 3174      | 3211      | 3167       | 1711       |
| Manager 2  | 1700      | 2670      | 1883      | 2534      | 1429      | 1173      | 777       | 248       | 1704      | 2603       | 1822       |
| Manager 3  | 160       | 755       | 3477      | 3122      | 2968      | 3023      | 1417      | 254       | 3175      | 2502       | 2595       |
| Manager 4  | 2213      | 1008      | 411       | 1199      | 418       | 1000      | 3148      | 1724      | 1984      | 1954       | 1805       |
| Manager 5  | 198       | 1721      | 1318      | 3194      | 3036      | 2938      | 3298      | 3332      | 1806      | 270        | 1893       |
| Manager 6  | 2375      | 1804      | 3174      | 1607      | 2168      | 1642      | 970       | 3433      | 1528      | 2696       | 2217       |
| Manager 7  | 2400      | 211       | 1172      | 425       | 1222      | 287       | 653       | 1466      | 479       | 2762       | 577        |
| Manager 8  | 272       | 2574      | 413       | 202       | 1220      | 2392      | 410       | 2250      | 2272      | 3260       | 2981       |
| Manager 9  | 2844      | 2775      | 357       | 2601      | 1627      | 125       | 1029      | 1354      | 2280      | 114        | 2161       |
| Manager 10 | 1222      | 296       | 3375      | 352       | 2167      | 2202      | 3139      | 2526      | 767       | 1873       | 1185       |
| Manager 11 | 2661      | 887       | 455       | 2552      | 1067      | 552       | 2991      | 1727      | 1639      | 3003       | 2161       |

##### Decision Variables

For each manager $i \in M$ and project $j \in P$:
$$
x_{ij} = \begin{cases}
1 & \text{if manager } i \text{ is assigned to project } j \\
0 & \text{otherwise}
\end{cases}
$$

##### Objective

Minimize the total assignment cost:
$$
\min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
$$

##### Constraints

1. Each manager is assigned to exactly one project:
$$
\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M
$$

2. Each project is assigned to exactly one manager:
$$
\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P
$$

3. Binary assignment variables:
$$
x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in P
$$

##### Complete Numerical Model

Minimize
\[
708x_{\text{Manager 1},\text{Project 1}} + 1948x_{\text{Manager 1},\text{Project 2}} + 2424x_{\text{Manager 1},\text{Project 3}} + 1068x_{\text{Manager 1},\text{Project 4}} + 729x_{\text{Manager 1},\text{Project 5}} + 199x_{\text{Manager 1},\text{Project 6}} + 1651x_{\text{Manager 1},\text{Project 7}} + 3174x_{\text{Manager 1},\text{Project 8}} + 3211x_{\text{Manager 1},\text{Project 9}} + 3167x_{\text{Manager 1},\text{Project 10}} + 1711x_{\text{Manager 1},\text{Project 11}}
\]
\[
+ 1700x_{\text{Manager 2},\text{Project 1}} + 2670x_{\text{Manager 2},\text{Project 2}} + 1883x_{\text{Manager 2},\text{Project 3}} + 2534x_{\text{Manager 2},\text{Project 4}} + 1429x_{\text{Manager 2},\text{Project 5}} + 1173x_{\text{Manager 2},\text{Project 6}} + 777x_{\text{Manager 2},\text{Project 7}} + 248x_{\text{Manager 2},\text{Project 8}} + 1704x_{\text{Manager 2},\text{Project 9}} + 2603x_{\text{Manager 2},\text{Project 10}} + 1822x_{\text{Manager 2},\text{Project 11}}
\]
\[
+ 160x_{\text{Manager 3},\text{Project 1}} + 755x_{\text{Manager 3},\text{Project 2}} + 3477x_{\text{Manager 3},\text{Project 3}} + 3122x_{\text{Manager 3},\text{Project 4}} + 2968x_{\text{Manager 3},\text{Project 5}} + 3023x_{\text{Manager 3},\text{Project 6}} + 1417x_{\text{Manager 3},\text{Project 7}} + 254x_{\text{Manager 3},\text{Project 8}} + 3175x_{\text{Manager 3},\text{Project 9}} + 2502x_{\text{Manager 3},\text{Project 10}} + 2595x_{\text{Manager 3},\text{Project 11}}
\]
\[
+ 2213x_{\text{Manager 4},\text{Project 1}} + 1008x_{\text{Manager 4},\text{Project 2}} + 411x_{\text{Manager 4},\text{Project 3}} + 1199x_{\text{Manager 4},\text{Project 4}} + 418x_{\text{Manager 4},\text{Project 5}} + 1000x_{\text{Manager 4},\text{Project 6}} + 3148x_{\text{Manager 4},\text{Project 7}} + 1724x_{\text{Manager 4},\text{Project 8}} + 1984x_{\text{Manager 4},\text{Project 9}} + 1954x_{\text{Manager 4},\text{Project 10}} + 1805x_{\text{Manager 4},\text{Project 11}}
\]
\[
+ 198x_{\text{Manager 5},\text{Project 1}} + 1721x_{\text{Manager 5},\text{Project 2}} + 1318x_{\text{Manager 5},\text{Project 3}} + 3194x_{\text{Manager 5},\text{Project 4}} + 3036x_{\text{Manager 5},\text{Project 5}} + 2938x_{\text{Manager 5},\text{Project 6}} + 3298x_{\text{Manager 5},\text{Project 7}} + 3332x_{\text{Manager 5},\text{Project 8}} + 1806x_{\text{Manager 5},\text{Project 9}} + 270x_{\text{Manager 5},\text{Project 10}} + 1893x_{\text{Manager 5},\text{Project 11}}
\]
\[
+ 2375x_{\text{Manager 6},\text{Project 1}} + 1804x_{\text{Manager 6},\text{Project 2}} + 3174x_{\text{Manager 6},\text{Project 3}} + 1607x_{\text{Manager 6},\text{Project 4}} + 2168x_{\text{Manager 6},\text{Project 5}} + 1642x_{\text{Manager 6},\text{Project 6}} + 970x_{\text{Manager 6},\text{Project 7}} + 3433x_{\text{Manager 6},\text{Project 8}} + 1528x_{\text{Manager 6},\text{Project 9}} + 2696x_{\text{Manager 6},\text{Project 10}} + 2217x_{\text{Manager 6},\text{Project 11}}
\]
\[
+ 2400x_{\text{Manager 7},\text{Project 1}} + 211x_{\text{Manager 7},\text{Project 2}} + 1172x_{\text{Manager 7},\text{Project 3}} + 425x_{\text{Manager 7},\text{Project 4}} + 1222x_{\text{Manager 7},\text{Project 5}} + 287x_{\text{Manager 7},\text{Project 6}} + 653x_{\text{Manager 7},\text{Project 7}} + 1466x_{\text{Manager 7},\text{Project 8}} + 479x_{\text{Manager 7},\text{Project 9}} + 2762x_{\text{Manager 7},\text{Project 10}} + 577x_{\text{Manager 7},\text{Project 11}}
\]
\[
+ 272x_{\text{Manager 8},\text{Project 1}} + 2574x_{\text{Manager 8},\text{Project 2}} + 413x_{\text{Manager 8},\text{Project 3}} + 202x_{\text{Manager 8},\text{Project 4}} + 1220x_{\text{Manager 8},\text{Project 5}} + 2392x_{\text{Manager 8},\text{Project 6}} + 410x_{\text{Manager 8},\text{Project 7}} + 2250x_{\text{Manager 8},\text{Project 8}} + 2272x_{\text{Manager 8},\text{Project 9}} + 3260x_{\text{Manager 8},\text{Project 10}} + 2981x_{\text{Manager 8},\text{Project 11}}
\]
\[
+ 2844x_{\text{Manager 9},\text{Project 1}} + 2775x_{\text{Manager 9},\text{Project 2}} + 357x_{\text{Manager 9},\text{Project 3}} + 2601x_{\text{Manager 9},\text{Project 4}} + 1627x_{\text{Manager 9},\text{Project 5}} + 125x_{\text{Manager 9},\text{Project 6}} + 1029x_{\text{Manager 9},\text{Project 7}} + 1354x_{\text{Manager 9},\text{Project 8}} + 2280x_{\text{Manager 9},\text{Project 9}} + 114x_{\text{Manager 9},\text{Project 10}} + 2161x_{\text{Manager 9},\text{Project 11}}
\]
\[
+ 1222x_{\text{Manager 10},\text{Project 1}} + 296x_{\text{Manager 10},\text{Project 2}} + 3375x_{\text{Manager 10},\text{Project 3}} + 352x_{\text{Manager 10},\text{Project 4}} + 2167x_{\text{Manager 10},\text{Project 5}} + 2202x_{\text{Manager 10},\text{Project 6}} + 3139x_{\text{Manager 10},\text{Project 7}} + 2526x_{\text{Manager 10},\text{Project 8}} + 767x_{\text{Manager 10},\text{Project 9}} + 1873x_{\text{Manager 10},\text{Project 10}} + 1185x_{\text{Manager 10},\text{Project 11}}
\]
\[
+ 2661x_{\text{Manager 11},\text{Project 1}} + 887x_{\text{Manager 11},\text{Project 2}} + 455x_{\text{Manager 11},\text{Project 3}} + 2552x_{\text{Manager 11},\text{Project 4}} + 1067x_{\text{Manager 11},\text{Project 5}} + 552x_{\text{Manager 11},\text{Project 6}} + 2991x_{\text{Manager 11},\text{Project 7}} + 1727x_{\text{Manager 11},\text{Project 8}} + 1639x_{\text{Manager 11},\text{Project 9}} + 3003x_{\text{Manager 11},\text{Project 10}} + 2161x_{\text{Manager 11},\text{Project 11}}
\]

Subject to:

For each manager $i$:
\[
\sum_{j \in P} x_{ij} = 1
\]

For each project $j$:
\[
\sum_{i \in M} x_{ij} = 1
\]

\[
x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in P
\]