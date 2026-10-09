##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center (supplier) $i$ to customer group $j$.

Where:
- $i \in I = \{S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12\}$
- $j \in J = \{C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12\}$

##### Parameters

Customer Demands (from customer_demand.csv):

\[
\begin{align*}
d_{C1} &= 52 \\
d_{C2} &= 80 \\
d_{C3} &= 392 \\
d_{C4} &= 103 \\
d_{C5} &= 32 \\
d_{C6} &= 1426 \\
d_{C7} &= 1024 \\
d_{C8} &= 2736 \\
d_{C9} &= 1129 \\
d_{C10} &= 676 \\
d_{C11} &= 2631 \\
d_{C12} &= 31 \\
\end{align*}
\]

Supplier Capacities (from supply_capacity.csv):

\[
\begin{align*}
s_{S1} &= 58 \\
s_{S2} &= 32 \\
s_{S3} &= 6161 \\
s_{S4} &= 4 \\
s_{S5} &= 47 \\
s_{S6} &= 178 \\
s_{S7} &= 142 \\
s_{S8} &= 164 \\
s_{S9} &= 1011 \\
s_{S10} &= 6 \\
s_{S11} &= 7081 \\
s_{S12} &= 948 \\
\end{align*}
\]

Transportation Costs $c_{ij}$ (from transportation_costs.csv):

|        | C1         | C2         | C3         | C4         | C5         | C6         | C7         | C8         | C9         | C10        | C11        | C12        |
|--------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| S1     | 134.7288   | 72.3005    | 37.9760    | 611.8466   | 1650.3354  | 32.9704    | 34.9984    | 73.2521    | 165.7521   | 82.5204    | 1538.8302  | 2320.1146  |
| S2     | 23.8354    | 1128.6333  | 187.0546   | 1.6077     | 2227.7037  | 72.4469    | 12.6007    | 1078.7730  | 383.7221   | 82.8115    | 1702.1725  | 1925.8447  |
| S3     | 1138.4988  | 231.8983   | 44.1568    | 962.5482   | 980.1306   | 1107.5158  | 741.7073   | 136.7020   | 1182.5161  | 664.1847   | 36.7024    | 1405.0507  |
| S4     | 1043.7563  | 24.2070    | 1120.9890  | 1027.5438  | 893.4458   | 1244.2627  | 980.0298   | 513.5650   | 977.6322   | 642.5520   | 437.9557   | 76.1250    |
| S5     | 4.9399     | 1278.1816  | 549.7571   | 21.3355    | 98.3269    | 452.0903   | 595.5982   | 70.8403    | 0.0012     | 1783.0866  | 1619.6155  | 116.0850   |
| S6     | 2105.5399  | 1340.9771  | 2077.2747  | 2202.2515  | 20.5337    | 2514.0984  | 2393.2135  | 1197.7655  | 102.3244   | 788.4640   | 818.9532   | 17.2496    |
| S7     | 61.3618    | 113.2372   | 50.2311    | 1219.1951  | 869.8690   | 58.3718    | 957.3213   | 168.9965   | 1363.3427  | 519.3858   | 483.3346   | 1412.7653  |
| S8     | 1169.3641  | 1037.2171  | 732.3578   | 865.2256   | 1510.0296  | 780.8538   | 860.7972   | 935.9126   | 61.8250    | 72.7839    | 1479.1137  | 65.8746    |
| S9     | 7.9365     | 1357.5388  | 628.3002   | 25.2451    | 1760.0085  | 604.6584   | 696.6775   | 1586.4093  | 89.6801    | 1580.7263  | 1423.5615  | 2000.5638  |
| S10    | 1685.3410  | 437.3938   | 1568.8940  | 1486.9269  | 498.3754   | 1493.3860  | 70.1795    | 526.9999   | 1527.6607  | 2.6764     | 202.7267   | 45.9531    |
| S11    | 937.1094   | 903.0141   | 264.7209   | 21.8233    | 1661.1055  | 18.5935    | 372.0818   | 956.6956   | 42.9120    | 1274.3228  | 1574.7962  | 1826.4283  |
| S12    | 1685.7577  | 377.3257   | 1347.0163  | 1737.0200  | 23.6122    | 83.0852    | 1476.1258  | 530.0022   | 1782.8443  | 0.0105     | 11.1693    | 963.9942   |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

where $c_{ij}$ is the transportation cost from supplier $i$ to customer $j$ as given above.

##### Constraints

1. **Demand Satisfaction** (each customer group must receive at least its demand):

   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply Capacity** (each supplier cannot ship more than its capacity):

   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity**:

   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Complete Numerical Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{i,C1} \geq 52 \\
& \sum_{i \in I} x_{i,C2} \geq 80 \\
& \sum_{i \in I} x_{i,C3} \geq 392 \\
& \sum_{i \in I} x_{i,C4} \geq 103 \\
& \sum_{i \in I} x_{i,C5} \geq 32 \\
& \sum_{i \in I} x_{i,C6} \geq 1426 \\
& \sum_{i \in I} x_{i,C7} \geq 1024 \\
& \sum_{i \in I} x_{i,C8} \geq 2736 \\
& \sum_{i \in I} x_{i,C9} \geq 1129 \\
& \sum_{i \in I} x_{i,C10} \geq 676 \\
& \sum_{i \in I} x_{i,C11} \geq 2631 \\
& \sum_{i \in I} x_{i,C12} \geq 31 \\
& \sum_{j \in J} x_{S1,j} \leq 58 \\
& \sum_{j \in J} x_{S2,j} \leq 32 \\
& \sum_{j \in J} x_{S3,j} \leq 6161 \\
& \sum_{j \in J} x_{S4,j} \leq 4 \\
& \sum_{j \in J} x_{S5,j} \leq 47 \\
& \sum_{j \in J} x_{S6,j} \leq 178 \\
& \sum_{j \in J} x_{S7,j} \leq 142 \\
& \sum_{j \in J} x_{S8,j} \leq 164 \\
& \sum_{j \in J} x_{S9,j} \leq 1011 \\
& \sum_{j \in J} x_{S10,j} \leq 6 \\
& \sum_{j \in J} x_{S11,j} \leq 7081 \\
& \sum_{j \in J} x_{S12,j} \leq 948 \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J \\
\end{align*}
\]

All $c_{ij}$ coefficients are as given in the table above.

###### Retrieved Information

- Customer demands for C1–C12, supplier capacities for S1–S12, and all transportation costs $c_{ij}$ from S1–S12 to C1–C12, with all identifiers and coefficients preserved in source order.