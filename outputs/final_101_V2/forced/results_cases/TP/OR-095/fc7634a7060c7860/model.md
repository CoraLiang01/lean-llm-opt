Let $x_i\geq0$ be the number of units produced of Widget $i$ for $i=1,\ldots,141$ (with $x_3$ for Widget3).  
Let $y\geq0$ be the kilograms of CatalystX sold (at most 1500 kg).  
Let $z\geq0$ be the kilograms of CatalystX disposed.

Parameters (from product_resources.csv and resource_limits.csv, in source order):

- For each $i=1,\ldots,141$:
  - $a_i$ = labor hours per unit (LaborHours)
  - $b_i$ = Material A per unit (MaterialA)
  - $c_i$ = Material B per unit (MaterialB)
  - $p_i$ = base profit per unit (Profit)
- Resource limits:
  - Total labor hours: $L=5000$
  - Total Material A: $A=24000$
  - Total Material B: $B=15000$
- Widget3 produces 5 kg CatalystX per unit: $5x_3$ kg total.
- CatalystX sales: $y\leq1500$
- CatalystX disposal: $z\geq0$
- All CatalystX produced must be either sold or disposed: $y+z=5x_3$
- Sale price: $300$/kg; disposal cost: $200$/kg.

Objective:
\[
\max \left( \sum_{i=1}^{141} p_i x_i + 300y - 200z \right)
\]

Subject to:
\[
\sum_{i=1}^{141} a_i x_i \leq 5000
\]
\[
\sum_{i=1}^{141} b_i x_i \leq 24000
\]
\[
\sum_{i=1}^{141} c_i x_i \leq 15000
\]
\[
y \leq 1500
\]
\[
y + z = 5x_3
\]
\[
x_i \geq 0 \quad \forall i=1,\ldots,141
\]
\[
y \geq 0,\quad z \geq 0
\]

Where the coefficients $a_i$, $b_i$, $c_i$, $p_i$ are as follows (source order):

| $i$ | Product   | $a_i$ (LaborHours) | $b_i$ (MaterialA) | $c_i$ (MaterialB) | $p_i$ (Profit) |
|-----|-----------|--------------------|-------------------|-------------------|---------------|
| 1   | Widget1   | 1.6                | 24                | 14                | 525           |
| 2   | Widget2   | 2                  | 20                | 10                | 678           |
| 3   | Widget3   | 2.5                | 12                | 18                | 812           |
| 4   | Widget4   | 1.9                | 21                | 15                | 769           |
| 5   | Widget5   | 0.0                | 15                | 26                | 952           |
| 6   | Widget6   | 0.1                | 24                | 17                | 987           |
| 7   | Widget7   | 1.2                | 15                | 30                | 644           |
| 8   | Widget8   | 1.3                | 21                | 24                | 795           |
| 9   | Widget9   | 0.4                | 20                | 30                | 829           |
| 10  | Widget10  | 0.9                | 18                | 27                | 574           |
| 11  | Widget11  | 0.2                | 24                | 21                | 723           |
| 12  | Widget12  | 1.6                | 20                | 20                | 775           |
| 13  | Widget13  | 0.1                | 20                | 22                | 992           |
| 14  | Widget14  | 0.7                | 12                | 19                | 568           |
| 15  | Widget15  | 1.1                | 11                | 28                | 895           |
| 16  | Widget16  | 0.7                | 15                | 25                | 838           |
| 17  | Widget17  | 1.0                | 14                | 27                | 883           |
| 18  | Widget18  | 1.8                | 18                | 15                | 567           |
| 19  | Widget19  | 0.5                | 23                | 25                | 829           |
| 20  | Widget20  | 1.2                | 17                | 29                | 821           |
| 21  | Widget21  | 0.2                | 20                | 17                | 893           |
| 22  | Widget22  | 1.5                | 10                | 22                | 977           |
| 23  | Widget23  | 1.1                | 12                | 21                | 536           |
| 24  | Widget24  | 0.6                | 12                | 24                | 797           |
| 25  | Widget25  | 1.8                | 10                | 25                | 793           |
| 26  | Widget26  | 2.0                | 15                | 23                | 754           |
| 27  | Widget27  | 1.4                | 17                | 21                | 819           |
| 28  | Widget28  | 1.3                | 22                | 30                | 947           |
| 29  | Widget29  | 1.2                | 11                | 17                | 857           |
| 30  | Widget30  | 0.3                | 13                | 29                | 886           |
| 31  | Widget31  | 1.6                | 17                | 15                | 609           |
| 32  | Widget32  | 0.9                | 24                | 19                | 706           |
| 33  | Widget33  | 0.1                | 10                | 26                | 545           |
| 34  | Widget34  | 0.0                | 12                | 22                | 762           |
| 35  | Widget35  | 1.8                | 12                | 21                | 719           |
| 36  | Widget36  | 0.3                | 24                | 26                | 793           |
| 37  | Widget37  | 0.6                | 20                | 27                | 865           |
| 38  | Widget38  | 0.7                | 16                | 23                | 932           |
| 39  | Widget39  | 0.6                | 17                | 18                | 962           |
| 40  | Widget40  | 1.3                | 10                | 22                | 824           |
| 41  | Widget41  | 0.8                | 15                | 19                | 992           |
| 42  | Widget42  | 1.7                | 21                | 29                | 536           |
| 43  | Widget43  | 1.5                | 17                | 25                | 896           |
| 44  | Widget44  | 1.3                | 25                | 23                | 969           |
| 45  | Widget45  | 1.8                | 21                | 26                | 558           |
| 46  | Widget46  | 0.1                | 25                | 20                | 893           |
| 47  | Widget47  | 1.0                | 11                | 30                | 777           |
| 48  | Widget48  | 0.6                | 23                | 29                | 718           |
| 49  | Widget49  | 0.2                | 25                | 16                | 811           |
| 50  | Widget50  | 1.2                | 25                | 29                | 545           |
| 51  | Widget51  | 0.7                | 17                | 25                | 961           |
| 52  | Widget52  | 1.7                | 12                | 25                | 657           |
| 53  | Widget53  | 0.8                | 12                | 23                | 658           |
| 54  | Widget54  | 1.6                | 15                | 19                | 860           |
| 55  | Widget55  | 0.5                | 22                | 20                | 625           |
| 56  | Widget56  | 1.1                | 18                | 17                | 510           |
| 57  | Widget57  | 0.0                | 17                | 24                | 1000          |
| 58  | Widget58  | 0.2                | 22                | 17                | 544           |
| 59  | Widget59  | 1.7                | 23                | 17                | 547           |
| 60  | Widget60  | 0.5                | 18                | 27                | 961           |
| 61  | Widget61  | 1.6                | 16                | 23                | 689           |
| 62  | Widget62  | 1.1                | 12                | 16                | 545           |
| 63  | Widget63  | 0.5                | 13                | 16                | 897           |
| 64  | Widget64  | 0.1                | 13                | 29                | 881           |
| 65  | Widget65  | 0.6                | 25                | 24                | 972           |
| 66  | Widget66  | 1.9                | 18                | 17                | 860           |
| 67  | Widget67  | 0.7                | 23                | 15                | 881           |
| 68  | Widget68  | 1.3                | 25                | 15                | 892           |
| 69  | Widget69  | 1.5                | 22                | 27                | 797           |
| 70  | Widget70  | 0.4                | 14                | 21                | 767           |
| 71  | Widget71  | 1.1                | 14                | 22                | 794           |
| 72  | Widget72  | 0.3                | 15                | 30                | 899           |
| 73  | Widget73  | 0.6                | 10                | 26                | 504           |
| 74  | Widget74  | 1.5                | 10                | 26                | 731           |
| 75  | Widget75  | 0.2                | 23                | 19                | 929           |
| 76  | Widget76  | 0.3                | 22                | 19                | 872           |
| 77  | Widget77  | 1.0                | 14                | 25                | 631           |
| 78  | Widget78  | 1.3                | 14                | 21                | 681           |
| 79  | Widget79  | 2.0                | 22                | 28                | 587           |
| 80  | Widget80  | 1.8                | 22                | 21                | 829           |
| 81  | Widget81  | 1.2                | 12                | 29                | 734           |
| 82  | Widget82  | 1.1                | 12                | 19                | 648           |
| 83  | Widget83  | 0.3                | 22                | 26                | 738           |
| 84  | Widget84  | 1.4                | 23                | 19                | 648           |
| 85  | Widget85  | 1.1                | 11                | 30                | 557           |
| 86  | Widget86  | 1.6                | 23                | 28                | 960           |
| 87  | Widget87  | 1.6                | 14                | 15                | 907           |
| 88  | Widget88  | 1.5                | 17                | 24                | 531           |
| 89  | Widget89  | 0.0                | 23                | 21                | 887           |
| 90  | Widget90  | 0.6                | 13                | 21                | 675           |
| 91  | Widget91  | 0.2                | 25                | 20                | 766           |
| 92  | Widget92  | 0.5                | 15                | 18                | 861           |
| 93  | Widget93  | 1.4                | 24                | 20                | 841           |
| 94  | Widget94  | 1.6                | 17                | 28                | 618           |
| 95  | Widget95  | 1.8                | 23                | 17                | 574           |
| 96  | Widget96  | 0.5                | 13                | 18                | 748           |
| 97  | Widget97  | 1.6                | 20                | 21                | 898           |
| 98  | Widget98  | 1.4                | 16                | 18                | 520           |
| 99  | Widget99  | 0.3                | 11                | 19                | 675           |
|100  | Widget100 | 1.3                | 22                | 26                | 868           |
|101  | Widget101 | 1.5                | 18                | 19                | 698           |
|102  | Widget102 | 0.7                | 12                | 17                | 988           |
|103  | Widget103 | 0.8                | 12                | 29                | 550           |
|104  | Widget104 | 1.8                | 10                | 27                | 508           |
|105  | Widget105 | 1.7                | 24                | 22                | 682           |
|106  | Widget106 | 0.8                | 21                | 20                | 995           |
|107  | Widget107 | 0.3                | 25                | 30                | 656           |
|108  | Widget108 | 1.3                | 25                | 22                | 569           |
|109  | Widget109 | 0.9                | 21                | 18                | 779           |
|110  | Widget110 | 1.6                | 23                | 26                | 965           |
|111  | Widget111 | 0.4                | 23                | 27                | 675           |
|112  | Widget112 | 1.7                | 21                | 30                | 638           |
|113  | Widget113 | 1.5                | 20                | 26                | 888           |
|114  | Widget114 | 2.0                | 15                | 26                | 875           |
|115  | Widget115 | 0.9                | 24                | 24                | 552           |
|116  | Widget116 | 0.9                | 12                | 23                | 705           |
|117  | Widget117 | 0.1                | 15                | 28                | 863           |
|118  | Widget118 | 0.1                | 14                | 18                | 706           |
|119  | Widget119 | 0.3                | 13                | 25                | 663           |
|120  | Widget120 | 0.1                | 13                | 20                | 859           |
|121  | Widget121 | 1.3                | 24                | 17                | 809           |
|122  | Widget122 | 1.5                | 12                | 17                | 830           |
|123  | Widget123 | 1.7                | 19                | 29                | 991           |
|124  | Widget124 | 0.7                | 10                | 16                | 791           |
|125  | Widget125 | 0.6                | 13                | 22                | 934           |
|126  | Widget126 | 0.8                | 22                | 15                | 732           |
|127  | Widget127 | 0.4                | 22                | 17                | 989           |
|128  | Widget128 | 1.4                | 21                | 21                | 573           |
|129  | Widget129 | 0.1                | 16                | 23                | 720           |
|130  | Widget130 | 0.5                | 19                | 27                | 834           |
|131  | Widget131 | 0.1                | 23                | 18                | 577           |
|132  | Widget132 | 0.9                | 17                | 20                | 951           |
|133  | Widget133 | 1.1                | 19                | 30                | 882           |
|134  | Widget134 | 1.7                | 17                | 17                | 859           |
|135  | Widget135 | 1.3                | 22                | 17                | 568           |
|136  | Widget136 | 0.8                | 17                | 26                | 861           |
|137  | Widget137 | 1.7                | 13                | 17                | 875           |
|138  | Widget138 | 1.7                | 12                | 18                | 834           |
|139  | Widget139 | 1.6                | 12                | 23                | 836           |
|140  | Widget140 | 1.6                | 17                | 24                | 938           |
|141  | Widget141 | 1.2                | 11                | 16                | 593           |

All coefficients and identifiers are preserved in source order. No data is omitted or aggregated.