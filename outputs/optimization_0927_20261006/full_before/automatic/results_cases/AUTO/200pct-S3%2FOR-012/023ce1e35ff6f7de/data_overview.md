Here are all the CSV rows required to formulate the model, in source order and with exact identifiers and values preserved:

**capacity.csv**
```
capacity_previous_year,resource_id,previous_period_capacity,resource_capacity,capacity_next_year_planning_forecast,capacity_two_periods_ago
1545,1,1576,1336,1411,1228
1497,2,2023,1754,1449,1410
1335,3,1937,1617,1367,1373
1328,4,1056,1119,1013,1227
1558,5,1197,1410,1690,1644
672,6,689,627,712,546
628,7,599,748,784,604
1272,8,1386,1540,1418,1617
1368,9,1294,1292,1544,1183
1257,10,996,1138,1167,1363
```

**products.csv**
```
item_name,previous_period_unit_value,previous_period_memory_profile,two_periods_ago_listing_status,two_periods_ago_unit_value,previous_period_listing_status,item_value,previous_period_resource_requirement,resource_requirement
Racing,30,Standard,Queued,33,Queued,28,462,393
Sports,61,Compact,Listed,59,Listed,69,229,195
Action,23,Compact,Queued,17,Listed,20,157,192
Adventure,70,Extended,Queued,55,Listed,62,166,155
RPG,62,Extended,Paused,57,Listed,58,578,500
Shooter,10,Compact,Listed,13,Queued,11,176,156
Strategy,78,Standard,Listed,72,Listed,73,283,317
Simulation,45,Compact,Paused,45,Queued,43,615,694
Puzzle,30,Standard,Queued,29,Listed,28,812,751
Fighting,55,Standard,Queued,59,Queued,57,504,467
Platformer,75,Extended,Paused,107,Paused,92,938,796
Survival,72,Extended,Listed,57,Listed,66,160,146
Horror,16,Standard,Paused,12,Paused,14,320,269
Sandbox,58,Standard,Queued,47,Queued,49,209,246
MMO,11,Extended,Queued,11,Queued,12,579,652
```