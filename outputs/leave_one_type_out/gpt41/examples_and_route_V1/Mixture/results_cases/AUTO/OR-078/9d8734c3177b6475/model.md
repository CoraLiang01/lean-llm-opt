Let $x_k$ be the number of lots purchased of option $k$ (where $k$ indexes the rows in the table below, using the option column as the identifier). All $x_k$ are nonnegative integers.

Minimize total cost:
$$
\min \sum_{k} \text{cost\_per\_lot}_k \cdot x_k
$$

Subject to the demand constraint:
$$
\sum_{k} \text{gen\_per\_lot}_k \cdot x_k \geq 200
$$

And integrality:
$$
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k
$$

Where the data for each $k$ is:

| option           | tech        | gen_per_lot | cost_per_lot |
|------------------|------------|-------------|--------------|
| coal_001         | coal       | 45          | 37.62        |
| coal_002         | coal       | 55          | 44.43        |
| coal_003         | coal       | 45          | 34.76        |
| coal_004         | coal       | 40          | 33.17        |
| coal_005         | coal       | 50          | 40.83        |
| coal_006         | coal       | 40          | 33.5         |
| coal_007         | coal       | 55          | 42.73        |
| coal_008         | coal       | 45          | 34.86        |
| coal_009         | coal       | 45          | 36.09        |
| coal_010         | coal       | 50          | 39.16        |
| coal_011         | coal       | 50          | 38.56        |
| coal_012         | coal       | 45          | 35.52        |
| coal_013         | coal       | 50          | 41.14        |
| coal_014         | coal       | 45          | 36.05        |
| coal_015         | coal       | 50          | 38.19        |
| coal_043         | coal       | 45          | 35.37        |
| coal_044         | coal       | 55          | 44.61        |
| coal_045         | coal       | 60          | 47.87        |
| coal_046         | coal       | 40          | 32.68        |
| coal_047         | coal       | 55          | 44.27        |
| coal_048         | coal       | 55          | 43.97        |
| coal_049         | coal       | 50          | 39.71        |
| coal_050         | coal       | 40          | 30.75        |
| coal_051         | coal       | 40          | 32.44        |
| coal_091         | coal       | 45          | 34.61        |
| coal_092         | coal       | 60          | 49.81        |
| coal_093         | coal       | 45          | 36.58        |
| coal_094         | coal       | 55          | 44.24        |
| coal_095         | coal       | 50          | 38.97        |
| coal_096         | coal       | 40          | 33.27        |
| coal_097         | coal       | 60          | 48.64        |
| coal_098         | coal       | 45          | 35.46        |
| coal_099         | coal       | 55          | 45.75        |
| coal_100         | coal       | 60          | 49.34        |
| gas_001          | gas        | 30          | 28.75        |
| gas_002          | gas        | 25          | 26           |
| gas_003          | gas        | 30          | 28.53        |
| gas_004          | gas        | 25          | 25.41        |
| gas_005          | gas        | 25          | 24.15        |
| gas_006          | gas        | 30          | 30.58        |
| gas_007          | gas        | 30          | 29.17        |
| gas_008          | gas        | 35          | 34.08        |
| gas_009          | gas        | 30          | 30.74        |
| gas_010          | gas        | 30          | 31.05        |
| gas_011          | gas        | 30          | 30.2         |
| gas_012          | gas        | 25          | 24.67        |
| gas_013          | gas        | 25          | 24.36        |
| gas_014          | gas        | 35          | 34.63        |
| gas_015          | gas        | 35          | 35.46        |
| gas_016          | gas        | 35          | 35.01        |
| gas_017          | gas        | 30          | 29.98        |
| gas_018          | gas        | 25          | 25.56        |
| gas_019          | gas        | 25          | 23.81        |
| gas_020          | gas        | 30          | 29.03        |
| gas_021          | gas        | 35          | 36.59        |
| gas_022          | gas        | 35          | 34.55        |
| gas_023          | gas        | 25          | 26.07        |
| gas_024          | gas        | 30          | 31.4         |
| gas_025          | gas        | 35          | 36.24        |
| gas_026          | gas        | 30          | 29.66        |
| gas_027          | gas        | 35          | 34.36        |
| gas_028          | gas        | 25          | 25.14        |
| gas_029          | gas        | 35          | 35.69        |
| gas_030          | gas        | 30          | 28.79        |
| gas_031          | gas        | 30          | 31.47        |
| gas_032          | gas        | 25          | 25.05        |
| gas_033          | gas        | 35          | 35.84        |
| gas_034          | gas        | 30          | 30.61        |
| gas_035          | gas        | 30          | 29.38        |
| gas_036          | gas        | 35          | 36.09        |
| gas_037          | gas        | 35          | 36.45        |
| gas_038          | gas        | 30          | 30           |
| gas_039          | gas        | 35          | 35.52        |
| gas_040          | gas        | 30          | 30.89        |
| gas_041          | gas        | 35          | 34.43        |
| gas_042          | gas        | 30          | 28.78        |
| gas_043          | gas        | 30          | 28.61        |
| gas_044          | gas        | 30          | 30.13        |
| gas_045          | gas        | 25          | 25.23        |
| gas_062          | gas        | 30          | 29.34        |
| gas_063          | gas        | 35          | 35.83        |
| gas_064          | gas        | 30          | 30.34        |
| gas_065          | gas        | 30          | 29.24        |
| gas_066          | gas        | 30          | 30.77        |
| gas_067          | gas        | 25          | 24.04        |
| gas_096          | gas        | 35          | 34.69        |
| gas_097          | gas        | 30          | 30.83        |
| gas_098          | gas        | 30          | 31.29        |
| gas_099          | gas        | 35          | 34.75        |
| gas_100          | gas        | 35          | 35.89        |
| renewables_001   | renewables | 15          | 19.5         |
| renewables_002   | renewables | 20          | 25.82        |
| renewables_003   | renewables | 20          | 25.99        |
| renewables_004   | renewables | 20          | 23.78        |
| renewables_005   | renewables | 25          | 29.97        |
| renewables_021   | renewables | 15          | 19.66        |
| renewables_022   | renewables | 20          | 24.68        |
| renewables_023   | renewables | 25          | 32.65        |
| renewables_024   | renewables | 25          | 32.04        |
| renewables_025   | renewables | 20          | 23.96        |
| renewables_026   | renewables | 20          | 25.15        |
| renewables_027   | renewables | 20          | 26.02        |
| renewables_028   | renewables | 15          | 18.74        |
| renewables_029   | renewables | 15          | 18.69        |
| renewables_030   | renewables | 15          | 18.04        |
| renewables_031   | renewables | 15          | 19.03        |
| renewables_032   | renewables | 20          | 25.21        |
| renewables_033   | renewables | 25          | 30.86        |
| renewables_034   | renewables | 20          | 25.92        |
| renewables_035   | renewables | 20          | 26.16        |
| renewables_036   | renewables | 15          | 19.63        |
| renewables_037   | renewables | 15          | 19.48        |
| renewables_038   | renewables | 20          | 26.23        |
| renewables_039   | renewables | 15          | 18.85        |
| renewables_040   | renewables | 25          | 31.32        |
| renewables_041   | renewables | 20          | 25.49        |
| renewables_042   | renewables | 20          | 25.32        |
| renewables_043   | renewables | 20          | 26           |
| renewables_044   | renewables | 15          | 18.34        |
| renewables_045   | renewables | 25          | 32.47        |
| renewables_046   | renewables | 20          | 25.3         |
| renewables_047   | renewables | 20          | 24.22        |
| renewables_048   | renewables | 20          | 24.63        |
| renewables_049   | renewables | 20          | 23.94        |
| renewables_050   | renewables | 25          | 32.77        |
| renewables_051   | renewables | 20          | 25.09        |
| renewables_052   | renewables | 20          | 25.78        |
| renewables_053   | renewables | 20          | 24.16        |
| renewables_054   | renewables | 25          | 32.26        |
| renewables_055   | renewables | 25          | 31.96        |
| renewables_056   | renewables | 20          | 24.8         |

All variables $x_k$ are nonnegative integers. The objective and constraint coefficients are as shown above. The total generation must be at least 200. The objective is to minimize total cost.