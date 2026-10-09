##### Sets and Indices

Let $i$ index the areas (as given by the ProductName column in products.csv).

##### Parameters

- $v_i$: Value (development benefit) for area $i$ (from products.csv, column Value)
- $w_i$: Weight (resource requirement) for area $i$ (from products.csv, column Weight)
- $C$: Capacity (overall development capacity, from capacity.csv, column Capacity)

##### Decision Variables

- $x_i$: Scale of development per day in area $i$ (continuous, $x_i \geq 0$)

##### Mathematical Model

Objective:
\[
\max \sum_{i} v_i x_i
\]

Subject to:
\[
\sum_{i} w_i x_i \leq C
\]
\[
x_i \geq 0 \quad \forall i
\]

##### Data

- Areas ($i$): 
    - Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, Williamsburg, Bushwick, Flatbush, Greenpoint, Park Slope, Astoria, Jackson Heights, Flushing, Sunnyside, Ditmars

- $v_i$ (Value):
    - Queens: 469
    - Brooklyn: 290
    - Manhattan: 236
    - Bronx: 235
    - Staten Island: 745
    - Harlem: 684
    - Upper East Side: 444
    - Lower Manhattan: 172
    - Midtown: 1000
    - Long Island City: 336
    - Williamsburg: 546
    - Bushwick: 535
    - Flatbush: 539
    - Greenpoint: 831
    - Park Slope: 139
    - Astoria: 432
    - Jackson Heights: 627
    - Flushing: 629
    - Sunnyside: 292
    - Ditmars: 978

- $w_i$ (Weight):
    - Queens: 954
    - Brooklyn: 650
    - Manhattan: 961
    - Bronx: 950
    - Staten Island: 379
    - Harlem: 776
    - Upper East Side: 381
    - Lower Manhattan: 808
    - Midtown: 937
    - Long Island City: 608
    - Williamsburg: 912
    - Bushwick: 391
    - Flatbush: 465
    - Greenpoint: 490
    - Park Slope: 918
    - Astoria: 787
    - Jackson Heights: 347
    - Flushing: 274
    - Sunnyside: 642
    - Ditmars: 130

- $C$ (Capacity): 586

##### Complete Model

\[
\begin{align*}
\max \quad & 469x_{\text{Queens}} + 290x_{\text{Brooklyn}} + 236x_{\text{Manhattan}} + 235x_{\text{Bronx}} + 745x_{\text{Staten Island}} + 684x_{\text{Harlem}} + 444x_{\text{Upper East Side}} + 172x_{\text{Lower Manhattan}} \\
& + 1000x_{\text{Midtown}} + 336x_{\text{Long Island City}} + 546x_{\text{Williamsburg}} + 535x_{\text{Bushwick}} + 539x_{\text{Flatbush}} + 831x_{\text{Greenpoint}} + 139x_{\text{Park Slope}} \\
& + 432x_{\text{Astoria}} + 627x_{\text{Jackson Heights}} + 629x_{\text{Flushing}} + 292x_{\text{Sunnyside}} + 978x_{\text{Ditmars}} \\
\text{s.t.} \quad & 954x_{\text{Queens}} + 650x_{\text{Brooklyn}} + 961x_{\text{Manhattan}} + 950x_{\text{Bronx}} + 379x_{\text{Staten Island}} + 776x_{\text{Harlem}} + 381x_{\text{Upper East Side}} + 808x_{\text{Lower Manhattan}} \\
& + 937x_{\text{Midtown}} + 608x_{\text{Long Island City}} + 912x_{\text{Williamsburg}} + 391x_{\text{Bushwick}} + 465x_{\text{Flatbush}} + 490x_{\text{Greenpoint}} + 918x_{\text{Park Slope}} \\
& + 787x_{\text{Astoria}} + 347x_{\text{Jackson Heights}} + 274x_{\text{Flushing}} + 642x_{\text{Sunnyside}} + 130x_{\text{Ditmars}} \leq 586 \\
& x_i \geq 0 \quad \forall i
\end{align*}
\]