##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

Where:
- $i \in I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $j \in J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Parameters

Customer group demands:
- $d_{\text{C1}} = 52$
- $d_{\text{C2}} = 80$
- $d_{\text{C3}} = 392$
- $d_{\text{C4}} = 103$
- $d_{\text{C5}} = 32$
- $d_{\text{C6}} = 1426$
- $d_{\text{C7}} = 1024$
- $d_{\text{C8}} = 2736$
- $d_{\text{C9}} = 1129$
- $d_{\text{C10}} = 676$
- $d_{\text{C11}} = 2631$
- $d_{\text{C12}} = 31$

Distribution center supply capacities:
- $s_{\text{S1}} = 58$
- $s_{\text{S2}} = 32$
- $s_{\text{S3}} = 6161$
- $s_{\text{S4}} = 4$
- $s_{\text{S5}} = 47$
- $s_{\text{S6}} = 178$
- $s_{\text{S7}} = 142$
- $s_{\text{S8}} = 164$
- $s_{\text{S9}} = 1011$
- $s_{\text{S10}} = 6$
- $s_{\text{S11}} = 7081$
- $s_{\text{S12}} = 948$

Transportation costs $c_{ij}$ (from distribution center $i$ to customer group $j$):

|        |  C1         |  C2         |  C3         |  C4         |  C5         |  C6         |  C7         |  C8         |  C9         |  C10        |  C11        |  C12        |
|--------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| S1     | 134.7288244 | 72.3004514  | 37.9759927  | 611.8465650 | 1650.3353902| 32.9704449  | 34.9983737  | 73.2520737  | 165.7520898 | 82.5203583  | 1538.8301984| 2320.1146477|
| S2     | 23.8353692  | 1128.633287 | 187.0545959 | 1.6077337   | 2227.7036991| 72.4469315  | 12.6007480  | 1078.7729548| 383.7220569 | 82.8115250  | 1702.1724764| 1925.8446971|
| S3     | 1138.4987596| 231.8983422 | 44.1568227  | 962.5481641 | 980.1306032 | 1107.5157507| 741.7073424 | 136.7020440 | 1182.5161115| 664.1846793 | 36.7023898  | 1405.0506919|
| S4     | 1043.7562803| 24.2070161  | 1120.9890439| 1027.5437639| 893.4457578 | 1244.2626779| 980.0297708 | 513.5650272 | 977.6321548 | 642.5520451 | 437.9556823 | 76.1249867  |
| S5     | 4.9399307   | 1278.1815735| 549.7570743 | 21.3354741  | 98.3268928  | 452.0903463 | 595.5981934 | 70.8402975  | 0.0012362   | 1783.0865677| 1619.6154656| 116.0849791 |
| S6     | 2105.5398596| 1340.9770560| 2077.2747284| 2202.2515396| 20.5336959  | 2514.0983598| 2393.2134665| 1197.7655434| 102.3244025 | 788.4640306 | 818.9532260 | 17.2496458  |
| S7     | 61.3618306  | 113.2371696 | 50.2310765  | 1219.1951077| 869.8689864 | 58.3717723  | 957.3213131 | 168.9964798 | 1363.3427136| 519.3858364 | 483.3345987 | 1412.7652882|
| S8     | 1169.3640989| 1037.2170745| 732.3577707 | 865.2255769 | 1510.0296187| 780.8537898 | 860.7971696 | 935.9126433 | 61.8250491  | 72.7839261  | 1479.1137195| 65.8746348  |
| S9     | 7.9365217   | 1357.5387610| 628.3001825 | 25.2451418  | 1760.0085250| 604.6583536 | 696.6774833 | 1586.4093184| 89.6800694  | 1580.7262556| 1423.5615412| 2000.5637997|
| S10    | 1685.3409759| 437.3938491 | 1568.8939936| 1486.9268968| 498.3754447 | 1493.3860214| 70.1795288  | 526.9999923 | 1527.6606597| 2.6764136   | 202.7267124 | 45.9530831  |
| S11    | 937.1094078 | 903.0141416 | 264.7208999 | 21.8233402  | 1661.1054863| 18.5934909  | 372.0817876 | 956.6955599 | 42.9120185  | 1274.3227527| 1574.7961500| 1826.4282573|
| S12    | 1685.7576851| 377.3256673 | 1347.0162725| 1737.0200305| 23.6121780  | 83.0852183  | 1476.1258242| 530.0021927 | 1782.8442633| 0.0104862   | 11.1693204  | 963.9941815 |

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** For each customer group $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   (Each customer group must receive at least its demand.)

2. **Supply capacity:** For each distribution center $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   (No distribution center may ship more than its supply capacity.)

3. **Non-negativity:** For all $i \in I$, $j \in J$,
   $$
   x_{ij} \geq 0
   $$

##### Sets and Indices

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Complete Model

$$
\begin{align*}
\min_{x_{ij} \geq 0} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
\end{align*}
$$

All parameters ($d_j$, $s_i$, $c_{ij}$) are as listed above, preserving all identifiers and coefficients from the supplied data.