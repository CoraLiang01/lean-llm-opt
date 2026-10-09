CSVQA_DATA = {'ignored_file_indices': [6],
 'query': 'Prepare the produce order for the CENTRAL_FRESH supermarket. Choose an integer number of cases of each '
          'product to maximize the net return from this order. The supplied tables describe the current plan; use '
          'their rows directly. The item rows give unit_benefit_cents and item_fee_cents: earn the former per unit and '
          'pay the latter once for any positive quantity. Use 1000 ml per liter, 60 minutes per hour and 1000 wh per '
          'kwh when comparing resource use with capacity. Only authorized=1 options can be ordered. Quantities are '
          'nonnegative integers: zero, or minimum_lot through maximum_order. For each resource, total per-unit usage '
          'times quantities must fit the signed sum of capacity_ledger entries. Meet every category minimum_quantity '
          'and maximum_quantity, and pay its activation_fee_cents once if any option is ordered. Do not select both '
          'members of an incompatible pair. An ordered item must have its requires prerequisite ordered as well; no '
          'quantity ratio applies. A bundle earns bonus_cents once only when both options have positive quantities; an '
          'unauthorized option cannot trigger a bonus. Report the maximum net benefit in USD cents. All fixed fees and '
          'bonuses are in the same unit.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['item_a', 'item_b', 'bonus_cents'],
             'file_index': 0,
             'file_name': 'export_01.csv',
             'filters': {},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'bonus_cents': '482', 'item_a': 'Rb3c64c386e4e', 'item_b': 'Ra083d004fdf3'}},
                         {'source_row': 1,
                          'values': {'bonus_cents': '576', 'item_a': 'Rc3f9ca031c8e', 'item_b': 'R7452a15d3717'}},
                         {'source_row': 2,
                          'values': {'bonus_cents': '502', 'item_a': 'Racf5f8a4fac2', 'item_b': 'Ref4615791cb9'}},
                         {'source_row': 3,
                          'values': {'bonus_cents': '186', 'item_a': 'R7afbee5ee47b', 'item_b': 'Rd501de06348a'}},
                         {'source_row': 4,
                          'values': {'bonus_cents': '127', 'item_a': 'Rc3f9ca031c8e', 'item_b': 'Ra04c52fef78e'}},
                         {'source_row': 5,
                          'values': {'bonus_cents': '463', 'item_a': 'R7afbee5ee47b', 'item_b': 'Rb3c64c386e4e'}},
                         {'source_row': 6,
                          'values': {'bonus_cents': '374', 'item_a': 'R7fbb3730d314', 'item_b': 'Ref4615791cb9'}},
                         {'source_row': 7,
                          'values': {'bonus_cents': '215', 'item_a': 'R5b0c501469fd', 'item_b': 'R363320090a95'}}],
             'returned_rows': 8,
             'role': 'bundle bonus',
             'table_id': 'file_0_view_0'},
            {'columns': ['resource', 'entry', 'amount', 'unit'],
             'file_index': 1,
             'file_name': 'export_02.csv',
             'filters': {},
             'original_rows': 6,
             'records': [{'source_row': 0,
                          'values': {'amount': '-5000', 'entry': 'reservation', 'resource': 'space', 'unit': 'ml'}},
                         {'source_row': 1,
                          'values': {'amount': '212000', 'entry': 'opening', 'resource': 'power', 'unit': 'wh'}},
                         {'source_row': 2,
                          'values': {'amount': '15540', 'entry': 'opening', 'resource': 'labor', 'unit': 'minute'}},
                         {'source_row': 3,
                          'values': {'amount': '-720', 'entry': 'reservation', 'resource': 'labor', 'unit': 'minute'}},
                         {'source_row': 4,
                          'values': {'amount': '224000', 'entry': 'opening', 'resource': 'space', 'unit': 'ml'}},
                         {'source_row': 5,
                          'values': {'amount': '-6000', 'entry': 'reservation', 'resource': 'power', 'unit': 'wh'}}],
             'returned_rows': 6,
             'role': 'resource capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents'],
             'file_index': 2,
             'file_name': 'export_03.csv',
             'filters': {},
             'original_rows': 4,
             'records': [{'source_row': 0,
                          'values': {'activation_fee_cents': '193',
                                     'category': 'G3',
                                     'maximum_quantity': '21',
                                     'minimum_quantity': '9'}},
                         {'source_row': 1,
                          'values': {'activation_fee_cents': '324',
                                     'category': 'G2',
                                     'maximum_quantity': '18',
                                     'minimum_quantity': '6'}},
                         {'source_row': 2,
                          'values': {'activation_fee_cents': '263',
                                     'category': 'G1',
                                     'maximum_quantity': '19',
                                     'minimum_quantity': '7'}},
                         {'source_row': 3,
                          'values': {'activation_fee_cents': '140',
                                     'category': 'G0',
                                     'maximum_quantity': '19',
                                     'minimum_quantity': '7'}}],
             'returned_rows': 4,
             'role': 'category constraints',
             'table_id': 'file_2_view_0'},
            {'columns': ['ref', 'kind', 'entity_id'],
             'file_index': 3,
             'file_name': 'export_04.csv',
             'filters': {},
             'original_rows': 24,
             'records': [{'source_row': 0, 'values': {'entity_id': 'I08', 'kind': 'item', 'ref': 'Ra04c52fef78e'}},
                         {'source_row': 1, 'values': {'entity_id': 'I00', 'kind': 'item', 'ref': 'Ra083d004fdf3'}},
                         {'source_row': 2, 'values': {'entity_id': 'I03', 'kind': 'item', 'ref': 'R7afbee5ee47b'}},
                         {'source_row': 3, 'values': {'entity_id': 'I17', 'kind': 'item', 'ref': 'Rb4a56a392c2a'}},
                         {'source_row': 4, 'values': {'entity_id': 'I04', 'kind': 'item', 'ref': 'Racf5f8a4fac2'}},
                         {'source_row': 5, 'values': {'entity_id': 'I16', 'kind': 'item', 'ref': 'Rc3f9ca031c8e'}},
                         {'source_row': 6, 'values': {'entity_id': 'I11', 'kind': 'item', 'ref': 'Rd501de06348a'}},
                         {'source_row': 7, 'values': {'entity_id': 'I23', 'kind': 'item', 'ref': 'R887ffaea87dd'}},
                         {'source_row': 8, 'values': {'entity_id': 'I15', 'kind': 'item', 'ref': 'R634e3af40d39'}},
                         {'source_row': 9, 'values': {'entity_id': 'I05', 'kind': 'item', 'ref': 'R7452a15d3717'}},
                         {'source_row': 10, 'values': {'entity_id': 'I18', 'kind': 'item', 'ref': 'R6a8f63724def'}},
                         {'source_row': 11, 'values': {'entity_id': 'I20', 'kind': 'item', 'ref': 'Ref4615791cb9'}},
                         {'source_row': 12, 'values': {'entity_id': 'I13', 'kind': 'item', 'ref': 'R7fbb3730d314'}},
                         {'source_row': 13, 'values': {'entity_id': 'I09', 'kind': 'item', 'ref': 'R363320090a95'}},
                         {'source_row': 14, 'values': {'entity_id': 'I07', 'kind': 'item', 'ref': 'Re4688318762d'}},
                         {'source_row': 15, 'values': {'entity_id': 'I19', 'kind': 'item', 'ref': 'Ra032aa43e993'}},
                         {'source_row': 16, 'values': {'entity_id': 'I12', 'kind': 'item', 'ref': 'R6cdd375c7a2a'}},
                         {'source_row': 17, 'values': {'entity_id': 'I01', 'kind': 'item', 'ref': 'R5b0c501469fd'}},
                         {'source_row': 18, 'values': {'entity_id': 'I06', 'kind': 'item', 'ref': 'Rb3c64c386e4e'}},
                         {'source_row': 19, 'values': {'entity_id': 'I02', 'kind': 'item', 'ref': 'R828da38f58bc'}},
                         {'source_row': 20, 'values': {'entity_id': 'I10', 'kind': 'item', 'ref': 'R863022fbf3f2'}},
                         {'source_row': 21, 'values': {'entity_id': 'I21', 'kind': 'item', 'ref': 'R35f4ccfb1574'}},
                         {'source_row': 22, 'values': {'entity_id': 'I14', 'kind': 'item', 'ref': 'Rbe86b9a56e99'}},
                         {'source_row': 23, 'values': {'entity_id': 'I22', 'kind': 'item', 'ref': 'R798a1de24f1f'}}],
             'returned_rows': 24,
             'role': 'item identity',
             'table_id': 'file_3_view_0'},
            {'columns': ['item_a', 'item_b'],
             'file_index': 4,
             'file_name': 'export_05.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'item_a': 'Rb4a56a392c2a', 'item_b': 'R7afbee5ee47b'}},
                         {'source_row': 1, 'values': {'item_a': 'Rd501de06348a', 'item_b': 'Rb4a56a392c2a'}},
                         {'source_row': 2, 'values': {'item_a': 'R6cdd375c7a2a', 'item_b': 'R7fbb3730d314'}},
                         {'source_row': 3, 'values': {'item_a': 'R887ffaea87dd', 'item_b': 'R6cdd375c7a2a'}},
                         {'source_row': 4, 'values': {'item_a': 'R828da38f58bc', 'item_b': 'R887ffaea87dd'}},
                         {'source_row': 5, 'values': {'item_a': 'Ref4615791cb9', 'item_b': 'R7fbb3730d314'}},
                         {'source_row': 6, 'values': {'item_a': 'R35f4ccfb1574', 'item_b': 'R6cdd375c7a2a'}},
                         {'source_row': 7, 'values': {'item_a': 'R863022fbf3f2', 'item_b': 'R7fbb3730d314'}},
                         {'source_row': 8, 'values': {'item_a': 'R7fbb3730d314', 'item_b': 'R828da38f58bc'}},
                         {'source_row': 9, 'values': {'item_a': 'R828da38f58bc', 'item_b': 'R6a8f63724def'}}],
             'returned_rows': 10,
             'role': 'incompatible items',
             'table_id': 'file_4_view_0'},
            {'columns': ['item_ref',
                         'category',
                         'authorized',
                         'minimum_lot',
                         'maximum_order',
                         'unit_benefit_cents',
                         'item_fee_cents'],
             'file_index': 5,
             'file_name': 'export_06.csv',
             'filters': {'conditions': [{'column': 'authorized',
                                         'dtype': 'number',
                                         'evidence': 'Only authorized=1 options can be ordered.',
                                         'operator': 'eq',
                                         'value': 1}],
                         'logic': 'and'},
             'original_rows': 24,
             'records': [{'source_row': 0,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'item_fee_cents': '431',
                                     'item_ref': 'R363320090a95',
                                     'maximum_order': '9',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '338'}},
                         {'source_row': 1,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'item_fee_cents': '155',
                                     'item_ref': 'R798a1de24f1f',
                                     'maximum_order': '7',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '482'}},
                         {'source_row': 2,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'item_fee_cents': '301',
                                     'item_ref': 'Rb3c64c386e4e',
                                     'maximum_order': '10',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '369'}},
                         {'source_row': 3,
                          'values': {'authorized': '1',
                                     'category': 'G0',
                                     'item_fee_cents': '432',
                                     'item_ref': 'Ra083d004fdf3',
                                     'maximum_order': '8',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '524'}},
                         {'source_row': 5,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'item_fee_cents': '258',
                                     'item_ref': 'Rd501de06348a',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '491'}},
                         {'source_row': 6,
                          'values': {'authorized': '1',
                                     'category': 'G0',
                                     'item_fee_cents': '240',
                                     'item_ref': 'Racf5f8a4fac2',
                                     'maximum_order': '11',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '581'}},
                         {'source_row': 7,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'item_fee_cents': '127',
                                     'item_ref': 'Re4688318762d',
                                     'maximum_order': '13',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '887'}},
                         {'source_row': 8,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'item_fee_cents': '184',
                                     'item_ref': 'Ra032aa43e993',
                                     'maximum_order': '7',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1425'}},
                         {'source_row': 9,
                          'values': {'authorized': '1',
                                     'category': 'G0',
                                     'item_fee_cents': '445',
                                     'item_ref': 'R6cdd375c7a2a',
                                     'maximum_order': '8',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1551'}},
                         {'source_row': 10,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'item_fee_cents': '314',
                                     'item_ref': 'Rb4a56a392c2a',
                                     'maximum_order': '9',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1518'}},
                         {'source_row': 11,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'item_fee_cents': '273',
                                     'item_ref': 'Rbe86b9a56e99',
                                     'maximum_order': '7',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1017'}},
                         {'source_row': 12,
                          'values': {'authorized': '1',
                                     'category': 'G0',
                                     'item_fee_cents': '308',
                                     'item_ref': 'Ra04c52fef78e',
                                     'maximum_order': '8',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '840'}},
                         {'source_row': 13,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'item_fee_cents': '184',
                                     'item_ref': 'R863022fbf3f2',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '372'}},
                         {'source_row': 14,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'item_fee_cents': '163',
                                     'item_ref': 'R634e3af40d39',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1427'}},
                         {'source_row': 16,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'item_fee_cents': '126',
                                     'item_ref': 'R6a8f63724def',
                                     'maximum_order': '12',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1181'}},
                         {'source_row': 17,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'item_fee_cents': '383',
                                     'item_ref': 'R7452a15d3717',
                                     'maximum_order': '9',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '852'}},
                         {'source_row': 18,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'item_fee_cents': '157',
                                     'item_ref': 'R5b0c501469fd',
                                     'maximum_order': '8',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '989'}},
                         {'source_row': 19,
                          'values': {'authorized': '1',
                                     'category': 'G3',
                                     'item_fee_cents': '116',
                                     'item_ref': 'R7afbee5ee47b',
                                     'maximum_order': '8',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1452'}},
                         {'source_row': 20,
                          'values': {'authorized': '1',
                                     'category': 'G1',
                                     'item_fee_cents': '191',
                                     'item_ref': 'R35f4ccfb1574',
                                     'maximum_order': '9',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '1416'}},
                         {'source_row': 22,
                          'values': {'authorized': '1',
                                     'category': 'G2',
                                     'item_fee_cents': '208',
                                     'item_ref': 'R828da38f58bc',
                                     'maximum_order': '11',
                                     'minimum_lot': '2',
                                     'unit_benefit_cents': '316'}}],
             'returned_rows': 20,
             'role': 'item order options',
             'table_id': 'file_5_view_0'},
            {'columns': ['item_ref', 'prerequisite_ref'],
             'file_index': 7,
             'file_name': 'export_08.csv',
             'filters': {},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'item_ref': 'R35f4ccfb1574', 'prerequisite_ref': 'Ra04c52fef78e'}},
                         {'source_row': 1,
                          'values': {'item_ref': 'Ra032aa43e993', 'prerequisite_ref': 'Racf5f8a4fac2'}},
                         {'source_row': 2,
                          'values': {'item_ref': 'R887ffaea87dd', 'prerequisite_ref': 'Ra04c52fef78e'}},
                         {'source_row': 3,
                          'values': {'item_ref': 'Rc3f9ca031c8e', 'prerequisite_ref': 'Re4688318762d'}},
                         {'source_row': 4,
                          'values': {'item_ref': 'Rbe86b9a56e99', 'prerequisite_ref': 'Ra083d004fdf3'}},
                         {'source_row': 5,
                          'values': {'item_ref': 'R6cdd375c7a2a', 'prerequisite_ref': 'Racf5f8a4fac2'}},
                         {'source_row': 6,
                          'values': {'item_ref': 'Rb4a56a392c2a', 'prerequisite_ref': 'R828da38f58bc'}},
                         {'source_row': 7,
                          'values': {'item_ref': 'R634e3af40d39', 'prerequisite_ref': 'Ra083d004fdf3'}},
                         {'source_row': 8,
                          'values': {'item_ref': 'R798a1de24f1f', 'prerequisite_ref': 'Racf5f8a4fac2'}},
                         {'source_row': 9,
                          'values': {'item_ref': 'Ref4615791cb9', 'prerequisite_ref': 'Rb3c64c386e4e'}},
                         {'source_row': 10,
                          'values': {'item_ref': 'R6a8f63724def', 'prerequisite_ref': 'R5b0c501469fd'}},
                         {'source_row': 11,
                          'values': {'item_ref': 'R7fbb3730d314', 'prerequisite_ref': 'R7afbee5ee47b'}}],
             'returned_rows': 12,
             'role': 'item prerequisites',
             'table_id': 'file_7_view_0'},
            {'columns': ['item_ref', 'resource', 'amount', 'unit'],
             'file_index': 8,
             'file_name': 'export_09.csv',
             'filters': {},
             'original_rows': 72,
             'records': [{'source_row': 0,
                          'values': {'amount': '4', 'item_ref': 'Rb4a56a392c2a', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 1,
                          'values': {'amount': '3', 'item_ref': 'Ra04c52fef78e', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 2,
                          'values': {'amount': '9', 'item_ref': 'Ra083d004fdf3', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 3,
                          'values': {'amount': '2', 'item_ref': 'Racf5f8a4fac2', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 4,
                          'values': {'amount': '8', 'item_ref': 'R798a1de24f1f', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 5,
                          'values': {'amount': '8', 'item_ref': 'R798a1de24f1f', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 6,
                          'values': {'amount': '6', 'item_ref': 'R35f4ccfb1574', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 7,
                          'values': {'amount': '4', 'item_ref': 'R363320090a95', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 8,
                          'values': {'amount': '4', 'item_ref': 'R863022fbf3f2', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 9,
                          'values': {'amount': '2', 'item_ref': 'R828da38f58bc', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 10,
                          'values': {'amount': '4', 'item_ref': 'Ra04c52fef78e', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 11,
                          'values': {'amount': '2', 'item_ref': 'R863022fbf3f2', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 12,
                          'values': {'amount': '3', 'item_ref': 'R7fbb3730d314', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 13,
                          'values': {'amount': '8', 'item_ref': 'R35f4ccfb1574', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 14,
                          'values': {'amount': '9', 'item_ref': 'R7452a15d3717', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 15,
                          'values': {'amount': '7', 'item_ref': 'R798a1de24f1f', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 16,
                          'values': {'amount': '6', 'item_ref': 'Ra083d004fdf3', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 17,
                          'values': {'amount': '2', 'item_ref': 'R6a8f63724def', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 18,
                          'values': {'amount': '4', 'item_ref': 'R5b0c501469fd', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 19,
                          'values': {'amount': '4', 'item_ref': 'R887ffaea87dd', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 20,
                          'values': {'amount': '7', 'item_ref': 'R5b0c501469fd', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 21,
                          'values': {'amount': '6', 'item_ref': 'Re4688318762d', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 22,
                          'values': {'amount': '2', 'item_ref': 'R363320090a95', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 23,
                          'values': {'amount': '9', 'item_ref': 'R863022fbf3f2', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 24,
                          'values': {'amount': '5', 'item_ref': 'Ra032aa43e993', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 25,
                          'values': {'amount': '9', 'item_ref': 'Racf5f8a4fac2', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 26,
                          'values': {'amount': '8', 'item_ref': 'R7afbee5ee47b', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 27,
                          'values': {'amount': '2', 'item_ref': 'Rb3c64c386e4e', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 28,
                          'values': {'amount': '5', 'item_ref': 'Rd501de06348a', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 29,
                          'values': {'amount': '6', 'item_ref': 'Re4688318762d', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 30,
                          'values': {'amount': '4', 'item_ref': 'R35f4ccfb1574', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 31,
                          'values': {'amount': '7', 'item_ref': 'R6a8f63724def', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 32,
                          'values': {'amount': '2', 'item_ref': 'Ra032aa43e993', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 33,
                          'values': {'amount': '7', 'item_ref': 'R6cdd375c7a2a', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 34,
                          'values': {'amount': '4', 'item_ref': 'Rb3c64c386e4e', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 35,
                          'values': {'amount': '5', 'item_ref': 'R6cdd375c7a2a', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 36,
                          'values': {'amount': '6', 'item_ref': 'Ref4615791cb9', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 37,
                          'values': {'amount': '3', 'item_ref': 'Rc3f9ca031c8e', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 38,
                          'values': {'amount': '7', 'item_ref': 'R363320090a95', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 39,
                          'values': {'amount': '8', 'item_ref': 'R7452a15d3717', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 40,
                          'values': {'amount': '7', 'item_ref': 'R7fbb3730d314', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 41,
                          'values': {'amount': '2', 'item_ref': 'Rb4a56a392c2a', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 42,
                          'values': {'amount': '7', 'item_ref': 'Ra04c52fef78e', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 43,
                          'values': {'amount': '2', 'item_ref': 'Racf5f8a4fac2', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 44,
                          'values': {'amount': '6', 'item_ref': 'Ra083d004fdf3', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 45,
                          'values': {'amount': '8', 'item_ref': 'R828da38f58bc', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 46,
                          'values': {'amount': '4', 'item_ref': 'R6a8f63724def', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 47,
                          'values': {'amount': '9', 'item_ref': 'R7fbb3730d314', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 48,
                          'values': {'amount': '4', 'item_ref': 'Ref4615791cb9', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 49,
                          'values': {'amount': '5', 'item_ref': 'R7afbee5ee47b', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 50,
                          'values': {'amount': '7', 'item_ref': 'R7452a15d3717', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 51,
                          'values': {'amount': '9', 'item_ref': 'Rb4a56a392c2a', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 52,
                          'values': {'amount': '2', 'item_ref': 'Rd501de06348a', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 53,
                          'values': {'amount': '9', 'item_ref': 'Ref4615791cb9', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 54,
                          'values': {'amount': '5', 'item_ref': 'R887ffaea87dd', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 55,
                          'values': {'amount': '7', 'item_ref': 'Rc3f9ca031c8e', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 56,
                          'values': {'amount': '9', 'item_ref': 'Rbe86b9a56e99', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 57,
                          'values': {'amount': '7', 'item_ref': 'R5b0c501469fd', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 58,
                          'values': {'amount': '8', 'item_ref': 'Rc3f9ca031c8e', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 59,
                          'values': {'amount': '7', 'item_ref': 'Rd501de06348a', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 60,
                          'values': {'amount': '6', 'item_ref': 'R634e3af40d39', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 61,
                          'values': {'amount': '4', 'item_ref': 'R887ffaea87dd', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 62,
                          'values': {'amount': '8', 'item_ref': 'R634e3af40d39', 'resource': 'space', 'unit': 'liter'}},
                         {'source_row': 63,
                          'values': {'amount': '6', 'item_ref': 'R6cdd375c7a2a', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 64,
                          'values': {'amount': '8', 'item_ref': 'R634e3af40d39', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 65,
                          'values': {'amount': '7', 'item_ref': 'R7afbee5ee47b', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 66,
                          'values': {'amount': '9', 'item_ref': 'Ra032aa43e993', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 67,
                          'values': {'amount': '4', 'item_ref': 'Rbe86b9a56e99', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 68,
                          'values': {'amount': '5', 'item_ref': 'Re4688318762d', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 69,
                          'values': {'amount': '6', 'item_ref': 'R828da38f58bc', 'resource': 'power', 'unit': 'kwh'}},
                         {'source_row': 70,
                          'values': {'amount': '7', 'item_ref': 'Rbe86b9a56e99', 'resource': 'labor', 'unit': 'hour'}},
                         {'source_row': 71,
                          'values': {'amount': '5', 'item_ref': 'Rb3c64c386e4e', 'resource': 'labor', 'unit': 'hour'}}],
             'returned_rows': 72,
             'role': 'item resource usage',
             'table_id': 'file_8_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem(CSVQA_FRAMES):
    items_df = CSVQA_FRAMES['file_5_view_0']
    items = []
    u_i = {}
    f_i = {}
    min_i = {}
    max_i = {}
    cat_of_i = {}
    for (idx, row) in items_df.iterrows():
        if str(row['authorized']).strip() != '1':
            continue
        i = row['item_ref']
        items.append(i)
        u_i[i] = float(row['unit_benefit_cents'])
        f_i[i] = float(row['item_fee_cents'])
        min_i[i] = int(row['minimum_lot'])
        max_i[i] = int(row['maximum_order'])
        cat_of_i[i] = row['category']
    cats_df = CSVQA_FRAMES['file_2_view_0']
    categories = []
    min_c = {}
    max_c = {}
    F_c = {}
    for (idx, row) in cats_df.iterrows():
        c = row['category']
        categories.append(c)
        min_c[c] = int(row['minimum_quantity'])
        max_c[c] = int(row['maximum_quantity'])
        F_c[c] = float(row['activation_fee_cents'])
    res_df = CSVQA_FRAMES['file_1_view_0']
    resource_units = {}
    cap_r = {}
    resource_entries = {}
    for (idx, row) in res_df.iterrows():
        r = row['resource']
        unit = row['unit'].strip().casefold()
        amt = float(row['amount'])
        if r not in resource_entries:
            resource_entries[r] = []
            resource_units[r] = unit
        resource_entries[r].append((row['entry'], amt, unit))
    for r in resource_entries:
        total = 0.0
        for (entry, amt, unit) in resource_entries[r]:
            if unit == 'liter':
                amt *= 1000
            elif unit == 'ml':
                pass
            elif unit == 'hour':
                amt *= 60
            elif unit == 'minute':
                pass
            elif unit == 'kwh':
                amt *= 1000
            elif unit == 'wh':
                pass
            else:
                raise ValueError(f'Unknown unit {unit} for resource {r}')
            total += amt
        cap_r[r] = total
    resources = list(cap_r.keys())
    usage_df = CSVQA_FRAMES['file_8_view_0']
    a_ir = {}
    for (idx, row) in usage_df.iterrows():
        i = row['item_ref']
        if i not in items:
            continue
        r = row['resource']
        amt = float(row['amount'])
        unit = row['unit'].strip().casefold()
        if unit == 'liter':
            amt *= 1000
        elif unit == 'ml':
            pass
        elif unit == 'hour':
            amt *= 60
        elif unit == 'minute':
            pass
        elif unit == 'kwh':
            amt *= 1000
        elif unit == 'wh':
            pass
        else:
            raise ValueError(f'Unknown unit {unit} for resource {r}')
        a_ir[i, r] = amt
    bundle_df = CSVQA_FRAMES['file_0_view_0']
    bundles = []
    bonus_ij = {}
    for (idx, row) in bundle_df.iterrows():
        i = row['item_a']
        j = row['item_b']
        if i in items and j in items:
            bundles.append((i, j))
            bonus_ij[i, j] = float(row['bonus_cents'])
    incomp_df = CSVQA_FRAMES['file_4_view_0']
    incomp_pairs = []
    for (idx, row) in incomp_df.iterrows():
        i = row['item_a']
        j = row['item_b']
        if i in items and j in items:
            incomp_pairs.append((i, j))
    prereq_df = CSVQA_FRAMES['file_7_view_0']
    prereq_pairs = []
    for (idx, row) in prereq_df.iterrows():
        i = row['item_ref']
        k = row['prerequisite_ref']
        if i in items and k in items:
            prereq_pairs.append((i, k))
    m = gp.Model('CENTRAL_FRESH_Produce_Order')
    x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    obj = gp.quicksum((u_i[i] * x_vars[i] for i in items))
    obj -= gp.quicksum((f_i[i] * z_vars[i] for i in items))
    obj += gp.quicksum((bonus_ij[i, j] * w_vars[i, j] for (i, j) in bundles))
    obj -= gp.quicksum((F_c[c] * y_vars[c] for c in categories))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    for i in items:
        m.addConstr(x_vars[i] >= min_i[i] * z_vars[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= max_i[i] * z_vars[i], name=f'maxorder_{i}')
    for c in categories:
        m.addConstr(gp.quicksum((x_vars[i] for i in items if cat_of_i[i] == c)) >= min_c[c], name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((x_vars[i] for i in items if cat_of_i[i] == c)) <= max_c[c], name=f'cat_max_{c}')
        for i in items:
            if cat_of_i[i] == c:
                m.addConstr(y_vars[c] >= z_vars[i], name=f'cat_act_{c}_{i}')
    for r in resources:
        m.addConstr(gp.quicksum((a_ir.get((i, r), 0.0) * x_vars[i] for i in items)) <= cap_r[r], name=f'rescap_{r}')
    for (i, j) in incomp_pairs:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incomp_{i}_{j}')
    for (i, k) in prereq_pairs:
        m.addConstr(z_vars[i] <= z_vars[k], name=f'prereq_{i}_{k}')
    for (i, j) in bundles:
        m.addConstr(w_vars[i, j] <= z_vars[i], name=f'bundle1_{i}_{j}')
        m.addConstr(w_vars[i, j] <= z_vars[j], name=f'bundle2_{i}_{j}')
        m.addConstr(w_vars[i, j] >= z_vars[i] + z_vars[j] - 1, name=f'bundle3_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)