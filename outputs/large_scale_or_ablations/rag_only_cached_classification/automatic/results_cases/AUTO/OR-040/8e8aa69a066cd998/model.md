Let the decision variables be:
x_1 = scale of development in Queens (integer, ≥ 0)
x_2 = scale of development in Brooklyn (integer, ≥ 0)
x_3 = scale of development in Manhattan (integer, ≥ 0)
x_4 = scale of development in Bronx (integer, ≥ 0)
x_5 = scale of development in Staten Island (integer, ≥ 0)
x_6 = scale of development in Harlem (integer, ≥ 0)
x_7 = scale of development in Upper East Side (integer, ≥ 0)
x_8 = scale of development in Lower Manhattan (integer, ≥ 0)
x_9 = scale of development in Midtown (integer, ≥ 0)
x_10 = scale of development in Long Island City (integer, ≥ 0)
x_11 = scale of development in Williamsburg (integer, ≥ 0)
x_12 = scale of development in Bushwick (integer, ≥ 0)
x_13 = scale of development in Flatbush (integer, ≥ 0)
x_14 = scale of development in Greenpoint (integer, ≥ 0)
x_15 = scale of development in Park Slope (integer, ≥ 0)
x_16 = scale of development in Astoria (integer, ≥ 0)
x_17 = scale of development in Jackson Heights (integer, ≥ 0)
x_18 = scale of development in Flushing (integer, ≥ 0)
x_19 = scale of development in Sunnyside (integer, ≥ 0)
x_20 = scale of development in Ditmars (integer, ≥ 0)

Maximize total benefit:
Maximize
 443 x_1 + 522 x_2 + 300 x_3 + 767 x_4 + 300 x_5 + 309 x_6 + 598 x_7 + 460 x_8 + 318 x_9 + 126 x_10
 + 593 x_11 + 871 x_12 + 858 x_13 + 321 x_14 + 275 x_15 + 700 x_16 + 685 x_17 + 940 x_18 + 522 x_19 + 763 x_20

Subject to the overall development-capacity constraint:
 104 x_1 + 368 x_2 + 483 x_3 + 165 x_4 + 105 x_5 + 123 x_6 + 131 x_7 + 341 x_8 + 258 x_9 + 469 x_10
 + 387 x_11 + 425 x_12 + 482 x_13 + 495 x_14 + 305 x_15 + 377 x_16 + 318 x_17 + 56 x_18 + 213 x_19 + 472 x_20 ≤ 4466

Variable domains:
 x_i ∈ {0, 1, 2, ...} for i = 1, ..., 20

Where the mapping of x_i to area is as listed above, in the original row order.