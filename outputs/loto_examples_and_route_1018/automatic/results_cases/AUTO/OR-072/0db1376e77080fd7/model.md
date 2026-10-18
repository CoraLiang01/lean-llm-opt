Let $x_t$ be the number of drivers and crew members who start work at the beginning of time period $t$ ($t=1,2,\ldots,24$; corresponding to Shift 1 to Shift 24 in the table below). Each person works continuously for 4 hours (i.e., covers 4 consecutive time periods).

Minimize the total number of drivers and crew members assigned:
$$
\min \sum_{t=1}^{24} x_t
$$

Subject to, for each time period $s=1,2,\ldots,24$ (Shift $s$), the sum of those who started in the current and previous three periods must meet or exceed the requirement in that period:

For $s=1$:
$$
x_1 + x_{24} + x_{23} + x_{22} \geq 20
$$

For $s=2$:
$$
x_2 + x_1 + x_{24} + x_{23} \geq 18
$$

For $s=3$:
$$
x_3 + x_2 + x_1 + x_{24} \geq 15
$$

For $s=4$:
$$
x_4 + x_3 + x_2 + x_1 \geq 15
$$

For $s=5$:
$$
x_5 + x_4 + x_3 + x_2 \geq 20
$$

For $s=6$:
$$
x_6 + x_5 + x_4 + x_3 \geq 30
$$

For $s=7$:
$$
x_7 + x_6 + x_5 + x_4 \geq 60
$$

For $s=8$:
$$
x_8 + x_7 + x_6 + x_5 \geq 70
$$

For $s=9$:
$$
x_9 + x_8 + x_7 + x_6 \geq 50
$$

For $s=10$:
$$
x_{10} + x_9 + x_8 + x_7 \geq 55
$$

For $s=11$:
$$
x_{11} + x_{10} + x_9 + x_8 \geq 65
$$

For $s=12$:
$$
x_{12} + x_{11} + x_{10} + x_9 \geq 75
$$

For $s=13$:
$$
x_{13} + x_{12} + x_{11} + x_{10} \geq 80
$$

For $s=14$:
$$
x_{14} + x_{13} + x_{12} + x_{11} \geq 70
$$

For $s=15$:
$$
x_{15} + x_{14} + x_{13} + x_{12} \geq 60
$$

For $s=16$:
$$
x_{16} + x_{15} + x_{14} + x_{13} \geq 55
$$

For $s=17$:
$$
x_{17} + x_{16} + x_{15} + x_{14} \geq 60
$$

For $s=18$:
$$
x_{18} + x_{17} + x_{16} + x_{15} \geq 75
$$

For $s=19$:
$$
x_{19} + x_{18} + x_{17} + x_{16} \geq 85
$$

For $s=20$:
$$
x_{20} + x_{19} + x_{18} + x_{17} \geq 70
$$

For $s=21$:
$$
x_{21} + x_{20} + x_{19} + x_{18} \geq 50
$$

For $s=22$:
$$
x_{22} + x_{21} + x_{20} + x_{19} \geq 40
$$

For $s=23$:
$$
x_{23} + x_{22} + x_{21} + x_{20} \geq 35
$$

For $s=24$:
$$
x_{24} + x_{23} + x_{22} + x_{21} \geq 25
$$

And non-negativity and integrality:
$$
x_t \in \mathbb{Z}_{\geq 0}, \quad t=1,2,\ldots,24
$$

Where the required numbers for each period are:

| Shift | Time         | Number Required |
|-------|--------------|----------------|
| 1     | 0:00-1:00    | 20             |
| 2     | 1:00-2:00    | 18             |
| 3     | 2:00-3:00    | 15             |
| 4     | 3:00-4:00    | 15             |
| 5     | 4:00-5:00    | 20             |
| 6     | 5:00-6:00    | 30             |
| 7     | 6:00-7:00    | 60             |
| 8     | 7:00-8:00    | 70             |
| 9     | 8:00-9:00    | 50             |
| 10    | 9:00-10:00   | 55             |
| 11    | 10:00-11:00  | 65             |
| 12    | 11:00-12:00  | 75             |
| 13    | 12:00-13:00  | 80             |
| 14    | 13:00-14:00  | 70             |
| 15    | 14:00-15:00  | 60             |
| 16    | 15:00-16:00  | 55             |
| 17    | 16:00-17:00  | 60             |
| 18    | 17:00-18:00  | 75             |
| 19    | 18:00-19:00  | 85             |
| 20    | 19:00-20:00  | 70             |
| 21    | 20:00-21:00  | 50             |
| 22    | 21:00-22:00  | 40             |
| 23    | 22:00-23:00  | 35             |
| 24    | 23:00-0:00   | 25             |