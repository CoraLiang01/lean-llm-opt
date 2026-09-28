##### Decision Variables

Let $x_k \in \mathbb{Z}_+$ be the number of lots purchased for generation option $k \in K$, where $K$ is the set of all generation options (coal, gas, renewables) as listed below.

##### Parameters

- $g_k$: generation per lot for option $k$
- $c_k$: cost per lot for option $k$
- $D = 200$: total demand to be met

All parameters are given below:

###### Coal Generation Options

| Identifier   | $g_k$ | $c_k$ |
|--------------|-------|-------|
| coal_001     | 45    | 37.62 |
| coal_002     | 55    | 44.43 |
| coal_003     | 45    | 34.76 |
| coal_004     | 40    | 33.17 |
| coal_005     | 50    | 40.83 |
| coal_006     | 40    | 33.5  |
| coal_007     | 55    | 42.73 |
| coal_008     | 45    | 34.86 |
| coal_009     | 45    | 36.09 |
| coal_010     | 50    | 39.16 |
| coal_011     | 50    | 38.56 |
| coal_012     | 45    | 35.52 |
| coal_013     | 50    | 41.14 |
| coal_014     | 45    | 36.05 |
| coal_015     | 50    | 38.19 |
| coal_043     | 45    | 35.37 |
| coal_044     | 55    | 44.61 |
| coal_045     | 60    | 47.87 |
| coal_046     | 40    | 32.68 |
| coal_047     | 55    | 44.27 |
| coal_048     | 55    | 43.97 |
| coal_049     | 50    | 39.71 |
| coal_050     | 40    | 30.75 |
| coal_051     | 40    | 32.44 |
| coal_091     | 45    | 34.61 |
| coal_092     | 60    | 49.81 |
| coal_093     | 45    | 36.58 |
| coal_094     | 55    | 44.24 |
| coal_095     | 50    | 38.97 |
| coal_096     | 40    | 33.27 |
| coal_097     | 60    | 48.64 |
| coal_098     | 45    | 35.46 |
| coal_099     | 55    | 45.75 |
| coal_100     | 60    | 49.34 |

###### Gas Generation Options

| Identifier   | $g_k$ | $c_k$ |
|--------------|-------|-------|
| gas_001      | 30    | 28.75 |
| gas_002      | 25    | 26    |
| gas_003      | 30    | 28.53 |
| gas_004      | 25    | 25.41 |
| gas_005      | 25    | 24.15 |
| gas_006      | 30    | 30.58 |
| gas_007      | 30    | 29.17 |
| gas_008      | 35    | 34.08 |
| gas_009      | 30    | 30.74 |
| gas_010      | 30    | 31.05 |
| gas_011      | 30    | 30.2  |
| gas_012      | 25    | 24.67 |
| gas_013      | 25    | 24.36 |
| gas_014      | 35    | 34.63 |
| gas_015      | 35    | 35.46 |
| gas_016      | 35    | 35.01 |
| gas_017      | 30    | 29.98 |
| gas_018      | 25    | 25.56 |
| gas_019      | 25    | 23.81 |
| gas_020      | 30    | 29.03 |
| gas_021      | 35    | 36.59 |
| gas_022      | 35    | 34.55 |
| gas_023      | 25    | 26.07 |
| gas_024      | 30    | 31.4  |
| gas_025      | 35    | 36.24 |
| gas_026      | 30    | 29.66 |
| gas_027      | 35    | 34.36 |
| gas_028      | 25    | 25.14 |
| gas_029      | 35    | 35.69 |
| gas_030      | 30    | 28.79 |
| gas_031      | 30    | 31.47 |
| gas_032      | 25    | 25.05 |
| gas_033      | 35    | 35.84 |
| gas_034      | 30    | 30.61 |
| gas_035      | 30    | 29.38 |
| gas_036      | 35    | 36.09 |
| gas_037      | 35    | 36.45 |
| gas_038      | 30    | 30    |
| gas_039      | 35    | 35.52 |
| gas_040      | 30    | 30.89 |
| gas_041      | 35    | 34.43 |
| gas_042      | 30    | 28.78 |
| gas_043      | 30    | 28.61 |
| gas_044      | 30    | 30.13 |
| gas_045      | 25    | 25.23 |
| gas_062      | 30    | 29.34 |
| gas_063      | 35    | 35.83 |
| gas_064      | 30    | 30.34 |
| gas_065      | 30    | 29.24 |
| gas_066      | 30    | 30.77 |
| gas_067      | 25    | 24.04 |
| gas_096      | 35    | 34.69 |
| gas_097      | 30    | 30.83 |
| gas_098      | 30    | 31.29 |
| gas_099      | 35    | 34.75 |
| gas_100      | 35    | 35.89 |

###### Renewables Generation Options

| Identifier        | $g_k$ | $c_k$ |
|-------------------|-------|-------|
| renewables_001    | 15    | 19.5  |
| renewables_002    | 20    | 25.82 |
| renewables_003    | 20    | 25.99 |
| renewables_004    | 20    | 23.78 |
| renewables_005    | 25    | 29.97 |
| renewables_021    | 15    | 19.66 |
| renewables_022    | 20    | 24.68 |
| renewables_023    | 25    | 32.65 |
| renewables_024    | 25    | 32.04 |
| renewables_025    | 20    | 23.96 |
| renewables_026    | 20    | 25.15 |
| renewables_027    | 20    | 26.02 |
| renewables_028    | 15    | 18.74 |
| renewables_029    | 15    | 18.69 |
| renewables_030    | 15    | 18.04 |
| renewables_031    | 15    | 19.03 |
| renewables_032    | 20    | 25.21 |
| renewables_033    | 25    | 30.86 |
| renewables_034    | 20    | 25.92 |
| renewables_035    | 20    | 26.16 |
| renewables_036    | 15    | 19.63 |
| renewables_037    | 15    | 19.48 |
| renewables_038    | 20    | 26.23 |
| renewables_039    | 15    | 18.85 |
| renewables_040    | 25    | 31.32 |
| renewables_041    | 20    | 25.49 |
| renewables_042    | 20    | 25.32 |
| renewables_043    | 20    | 26    |
| renewables_044    | 15    | 18.34 |
| renewables_045    | 25    | 32.47 |
| renewables_046    | 20    | 25.3  |
| renewables_047    | 20    | 24.22 |
| renewables_048    | 20    | 24.63 |
| renewables_049    | 20    | 23.94 |
| renewables_050    | 25    | 32.77 |
| renewables_051    | 20    | 25.09 |
| renewables_052    | 20    | 25.78 |
| renewables_053    | 20    | 24.16 |
| renewables_054    | 25    | 32.26 |
| renewables_055    | 25    | 31.96 |
| renewables_056    | 20    | 24.8  |

##### Objective Function

\[
\min \sum_{k \in K} c_k x_k
\]

##### Constraints

1. **Demand Satisfaction:**
   \[
   \sum_{k \in K} g_k x_k \geq D
   \]
   where $D = 200$.

2. **Lot Integrality:**
   \[
   x_k \in \mathbb{Z}_+, \quad \forall k \in K
   \]

##### Summary of Sets

- $K$ is the set of all generation options listed above (coal, gas, renewables), with their respective $g_k$ and $c_k$ as given.

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{k \in K} c_k x_k \\
\text{s.t.} \quad & \sum_{k \in K} g_k x_k \geq 200 \\
& x_k \in \mathbb{Z}_+, \quad \forall k \in K
\end{align*}
\]

All parameters ($g_k$, $c_k$) and identifiers are as listed above.