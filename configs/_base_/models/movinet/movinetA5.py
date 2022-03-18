"""
param list for MoViNetA5:
-------------------------------------------------------------------------------------------------------------
|  name  | input_channels | out_channels | expanded_channels | kernel_size | stride | padding | padding_avg | 
-------------------------------------------------------------------------------------------------------------
| conv1  |       3        |      24      |        -          |   (1,3,3)   | (1,2,2)| (0,1,1) |       -     |
-------------------------------------------------------------------------------------------------------------
|Block2_1|      24        |      24      |        64         |   (1,5,5)   | (1,2,2)| (0,1,1) |   (0,0,0)   |
-------------------------------------------------------------------------------------------------------------
|Block2_2|      24        |      24      |        64         |   (3,3,3)   | (1,1,1)| (1,1,1) |   (0,1,1)   |
-------------------------------------------------------------------------------------------------------------
|Block2_3|      24        |      24      |        96         |   (3,3,3)   | (1,2,2)| (0,1,1) |   (0,0,0)   |
-------------------------------------------------------------------------------------------------------------

"""

cfg = dict()
cfg['name'] = 'A5'
cfg['conv1'] = dict()
cfg['conv1']['input_channels'] = 3
cfg['conv1']['out_channels'] = 24
cfg['conv1']['kernel_size'] = (1,3,3)
cfg['conv1']['stride'] = (1,2,2)
cfg['conv1']['padding'] = (0,1,1)
# fill_conv(cfg['conv1'], 3, 24, (1,3,3), (1,2,2), (0,1,1))
cfg['blocks'] = [[dict() for _ in range(6)],
                 [dict() for _ in range(11)],
                 [dict() for _ in range(13)],
                 [dict() for _ in range(11)],
                 [dict() for _ in range(18)]]

# block 2
# cfg['blocks'][0][0], 24, 24, 64, (1,5,5), (1,2,2), (0,2,2), (0,1,1)
cfg['blocks'][0][0]['input_channels'] = 24
cfg['blocks'][0][0]['out_channels'] = 24
cfg['blocks'][0][0]['expanded_channels'] = 64
cfg['blocks'][0][0]['kernel_size'] = (1,5,5)
cfg['blocks'][0][0]['stride'] = (1,2,2)
cfg['blocks'][0][0]['padding'] = (0,2,2)
cfg['blocks'][0][0]['padding_avg'] = (0,1,1)
# cfg['blocks'][0][1], 24, 24, 64, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][0][1]['input_channels'] = 24
cfg['blocks'][0][1]['out_channels'] = 24
cfg['blocks'][0][1]['expanded_channels'] = 64
cfg['blocks'][0][1]['kernel_size'] = (1,5,5)
cfg['blocks'][0][1]['stride'] = (1,1,1)
cfg['blocks'][0][1]['padding'] = (0,2,2)
cfg['blocks'][0][1]['padding_avg'] = (0,1,1)
# cfg['blocks'][0][2], 24, 24, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][0][2]['input_channels'] = 24
cfg['blocks'][0][2]['out_channels'] = 24
cfg['blocks'][0][2]['expanded_channels'] = 96
cfg['blocks'][0][2]['kernel_size'] = (3,3,3)
cfg['blocks'][0][2]['stride'] = (1,1,1)
cfg['blocks'][0][2]['padding'] = (1,1,1)
cfg['blocks'][0][2]['padding_avg'] = (0,1,1)
# cfg['blocks'][0][3], 24, 24, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][0][3]['input_channels'] = 24
cfg['blocks'][0][3]['out_channels'] = 24
cfg['blocks'][0][3]['expanded_channels'] = 64
cfg['blocks'][0][3]['kernel_size'] = (3,3,3)
cfg['blocks'][0][3]['stride'] = (1,1,1)
cfg['blocks'][0][3]['padding'] = (1,1,1)
cfg['blocks'][0][3]['padding_avg'] = (0,1,1)
# cfg['blocks'][0][4], 24, 24, 96, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][0][4]['input_channels'] = 24
cfg['blocks'][0][4]['out_channels'] = 24
cfg['blocks'][0][4]['expanded_channels'] = 96
cfg['blocks'][0][4]['kernel_size'] = (3,3,3)
cfg['blocks'][0][4]['stride'] = (1,1,1)
cfg['blocks'][0][4]['padding'] = (1,1,1)
cfg['blocks'][0][4]['padding_avg'] = (0,1,1)
# cfg['blocks'][0][5], 24, 24, 64, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][0][5]['input_channels'] = 24
cfg['blocks'][0][5]['out_channels'] = 24
cfg['blocks'][0][5]['expanded_channels'] = 64
cfg['blocks'][0][5]['kernel_size'] = (3,3,3)
cfg['blocks'][0][5]['stride'] = (1,1,1)
cfg['blocks'][0][5]['padding'] = (1,1,1)
cfg['blocks'][0][5]['padding_avg'] = (0,1,1)

# block 3
# cfg['blocks'][1][0], 24, 64, 192, (5,3,3), (1,2,2), (2,1,1), (0,1,1)
cfg['blocks'][1][0]['input_channels'] = 24
cfg['blocks'][1][0]['out_channels'] = 64
cfg['blocks'][1][0]['expanded_channels'] = 192
cfg['blocks'][1][0]['kernel_size'] = (5,3,3)
cfg['blocks'][1][0]['stride'] = (1,2,2)
cfg['blocks'][1][0]['padding'] = (2,1,1)
cfg['blocks'][1][0]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][1], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][1]['input_channels'] = 64
cfg['blocks'][1][1]['out_channels'] = 64
cfg['blocks'][1][1]['expanded_channels'] = 152
cfg['blocks'][1][1]['kernel_size'] = (3,3,3)
cfg['blocks'][1][1]['stride'] = (1,1,1)
cfg['blocks'][1][1]['padding'] = (1,1,1)
cfg['blocks'][1][1]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][2], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][2]['input_channels'] = 64
cfg['blocks'][1][2]['out_channels'] = 64
cfg['blocks'][1][2]['expanded_channels'] = 152
cfg['blocks'][1][2]['kernel_size'] = (3,3,3)
cfg['blocks'][1][2]['stride'] = (1,1,1)
cfg['blocks'][1][2]['padding'] = (1,1,1)
cfg['blocks'][1][2]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][3], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][3]['input_channels'] = 64
cfg['blocks'][1][3]['out_channels'] = 64
cfg['blocks'][1][3]['expanded_channels'] = 152
cfg['blocks'][1][3]['kernel_size'] = (3,3,3)
cfg['blocks'][1][3]['stride'] = (1,1,1)
cfg['blocks'][1][3]['padding'] = (1,1,1)
cfg['blocks'][1][3]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][4], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][4]['input_channels'] = 64
cfg['blocks'][1][4]['out_channels'] = 64
cfg['blocks'][1][4]['expanded_channels'] = 192
cfg['blocks'][1][4]['kernel_size'] = (3,3,3)
cfg['blocks'][1][4]['stride'] = (1,1,1)
cfg['blocks'][1][4]['padding'] = (1,1,1)
cfg['blocks'][1][4]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][5], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][5]['input_channels'] = 64
cfg['blocks'][1][5]['out_channels'] = 64
cfg['blocks'][1][5]['expanded_channels'] = 192
cfg['blocks'][1][5]['kernel_size'] = (3,3,3)
cfg['blocks'][1][5]['stride'] = (1,1,1)
cfg['blocks'][1][5]['padding'] = (1,1,1)
cfg['blocks'][1][5]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][6], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][6]['input_channels'] = 64
cfg['blocks'][1][6]['out_channels'] = 64
cfg['blocks'][1][6]['expanded_channels'] = 192
cfg['blocks'][1][6]['kernel_size'] = (3,3,3)
cfg['blocks'][1][6]['stride'] = (1,1,1)
cfg['blocks'][1][6]['padding'] = (1,1,1)
cfg['blocks'][1][6]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][7], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][7]['input_channels'] = 64
cfg['blocks'][1][7]['out_channels'] = 64
cfg['blocks'][1][7]['expanded_channels'] = 152
cfg['blocks'][1][7]['kernel_size'] = (3,3,3)
cfg['blocks'][1][7]['stride'] = (1,1,1)
cfg['blocks'][1][7]['padding'] = (1,1,1)
cfg['blocks'][1][7]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][8], 64, 64, 152, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][8]['input_channels'] = 64
cfg['blocks'][1][8]['out_channels'] = 64
cfg['blocks'][1][8]['expanded_channels'] = 152
cfg['blocks'][1][8]['kernel_size'] = (3,3,3)
cfg['blocks'][1][8]['stride'] = (1,1,1)
cfg['blocks'][1][8]['padding'] = (1,1,1)
cfg['blocks'][1][8]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][9], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][9]['input_channels'] = 64
cfg['blocks'][1][9]['out_channels'] = 64
cfg['blocks'][1][9]['expanded_channels'] = 192
cfg['blocks'][1][9]['kernel_size'] = (3,3,3)
cfg['blocks'][1][9]['stride'] = (1,1,1)
cfg['blocks'][1][9]['padding'] = (1,1,1)
cfg['blocks'][1][9]['padding_avg'] = (0,1,1)
# cfg['blocks'][1][10], 64, 64, 192, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][1][10]['input_channels'] = 64
cfg['blocks'][1][10]['out_channels'] = 64
cfg['blocks'][1][10]['expanded_channels'] = 192
cfg['blocks'][1][10]['kernel_size'] = (3,3,3)
cfg['blocks'][1][10]['stride'] = (1,1,1)
cfg['blocks'][1][10]['padding'] = (1,1,1)
cfg['blocks'][1][10]['padding_avg'] = (0,1,1)


# block 4
# cfg['blocks'][2][0], 64, 112, 376, (5,3,3), (1,2,2), (2,1,1), (0,1,1)
cfg['blocks'][2][0]['input_channels'] = 64
cfg['blocks'][2][0]['out_channels'] = 112
cfg['blocks'][2][0]['expanded_channels'] = 376
cfg['blocks'][2][0]['kernel_size'] = (5,3,3)
cfg['blocks'][2][0]['stride'] = (1,2,2)
cfg['blocks'][2][0]['padding'] = (2,1,1)
cfg['blocks'][2][0]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][1], 112, 112, 224, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][1]['input_channels'] = 112
cfg['blocks'][2][1]['out_channels'] = 112
cfg['blocks'][2][1]['expanded_channels'] = 224
cfg['blocks'][2][1]['kernel_size'] = (3,3,3)
cfg['blocks'][2][1]['stride'] = (1,1,1)
cfg['blocks'][2][1]['padding'] = (1,1,1)
cfg['blocks'][2][1]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][2], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][2]['input_channels'] = 112
cfg['blocks'][2][2]['out_channels'] = 112
cfg['blocks'][2][2]['expanded_channels'] = 376
cfg['blocks'][2][2]['kernel_size'] = (3,3,3)
cfg['blocks'][2][2]['stride'] = (1,1,1)
cfg['blocks'][2][2]['padding'] = (1,1,1)
cfg['blocks'][2][2]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][3], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][3]['input_channels'] = 112
cfg['blocks'][2][3]['out_channels'] = 112
cfg['blocks'][2][3]['expanded_channels'] = 376
cfg['blocks'][2][3]['kernel_size'] = (3,3,3)
cfg['blocks'][2][3]['stride'] = (1,1,1)
cfg['blocks'][2][3]['padding'] = (1,1,1)
cfg['blocks'][2][3]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][4], 112, 112, 296, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][4]['input_channels'] = 112
cfg['blocks'][2][4]['out_channels'] = 112
cfg['blocks'][2][4]['expanded_channels'] = 296
cfg['blocks'][2][4]['kernel_size'] = (3,3,3)
cfg['blocks'][2][4]['stride'] = (1,1,1)
cfg['blocks'][2][4]['padding'] = (1,1,1)
cfg['blocks'][2][4]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][5], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][5]['input_channels'] = 112
cfg['blocks'][2][5]['out_channels'] = 112
cfg['blocks'][2][5]['expanded_channels'] = 376
cfg['blocks'][2][5]['kernel_size'] = (3,3,3)
cfg['blocks'][2][5]['stride'] = (1,1,1)
cfg['blocks'][2][5]['padding'] = (1,1,1)
cfg['blocks'][2][5]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][6], 112, 112, 224, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][6]['input_channels'] = 112
cfg['blocks'][2][6]['out_channels'] = 112
cfg['blocks'][2][6]['expanded_channels'] = 224
cfg['blocks'][2][6]['kernel_size'] = (3,3,3)
cfg['blocks'][2][6]['stride'] = (1,1,1)
cfg['blocks'][2][6]['padding'] = (1,1,1)
cfg['blocks'][2][6]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][7], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][7]['input_channels'] = 112
cfg['blocks'][2][7]['out_channels'] = 112
cfg['blocks'][2][7]['expanded_channels'] = 376
cfg['blocks'][2][7]['kernel_size'] = (3,3,3)
cfg['blocks'][2][7]['stride'] = (1,1,1)
cfg['blocks'][2][7]['padding'] = (1,1,1)
cfg['blocks'][2][7]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][8], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][8]['input_channels'] = 112
cfg['blocks'][2][8]['out_channels'] = 112
cfg['blocks'][2][8]['expanded_channels'] = 376
cfg['blocks'][2][8]['kernel_size'] = (3,3,3)
cfg['blocks'][2][8]['stride'] = (1,1,1)
cfg['blocks'][2][8]['padding'] = (1,1,1)
cfg['blocks'][2][8]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][9], 112, 112, 296, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][9]['input_channels'] = 112
cfg['blocks'][2][9]['out_channels'] = 112
cfg['blocks'][2][9]['expanded_channels'] = 296
cfg['blocks'][2][9]['kernel_size'] = (3,3,3)
cfg['blocks'][2][9]['stride'] = (1,1,1)
cfg['blocks'][2][9]['padding'] = (1,1,1)
cfg['blocks'][2][9]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][10], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][10]['input_channels'] = 112
cfg['blocks'][2][10]['out_channels'] = 112
cfg['blocks'][2][10]['expanded_channels'] = 376
cfg['blocks'][2][10]['kernel_size'] = (3,3,3)
cfg['blocks'][2][10]['stride'] = (1,1,1)
cfg['blocks'][2][10]['padding'] = (1,1,1)
cfg['blocks'][2][10]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][11], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][11]['input_channels'] = 112
cfg['blocks'][2][11]['out_channels'] = 112
cfg['blocks'][2][11]['expanded_channels'] = 376
cfg['blocks'][2][11]['kernel_size'] = (3,3,3)
cfg['blocks'][2][11]['stride'] = (1,1,1)
cfg['blocks'][2][11]['padding'] = (1,1,1)
cfg['blocks'][2][11]['padding_avg'] = (0,1,1)
# cfg['blocks'][2][12], 112, 112, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][2][12]['input_channels'] = 112
cfg['blocks'][2][12]['out_channels'] = 112
cfg['blocks'][2][12]['expanded_channels'] = 376
cfg['blocks'][2][12]['kernel_size'] = (3,3,3)
cfg['blocks'][2][12]['stride'] = (1,1,1)
cfg['blocks'][2][12]['padding'] = (1,1,1)
cfg['blocks'][2][12]['padding_avg'] = (0,1,1)


# block 5
# cfg['blocks'][3][0], 112, 120, 376, (5,3,3), (1,1,1), (2,1,1), (0,1,1)
cfg['blocks'][3][0]['input_channels'] = 112
cfg['blocks'][3][0]['out_channels'] = 120
cfg['blocks'][3][0]['expanded_channels'] = 376
cfg['blocks'][3][0]['kernel_size'] = (5,3,3)
cfg['blocks'][3][0]['stride'] = (1,1,1)
cfg['blocks'][3][0]['padding'] = (2,1,1)
cfg['blocks'][3][0]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][1], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][1]['input_channels'] = 120
cfg['blocks'][3][1]['out_channels'] = 120
cfg['blocks'][3][1]['expanded_channels'] = 376
cfg['blocks'][3][1]['kernel_size'] = (3,3,3)
cfg['blocks'][3][1]['stride'] = (1,1,1)
cfg['blocks'][3][1]['padding'] = (1,1,1)
cfg['blocks'][3][1]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][2], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][2]['input_channels'] = 120
cfg['blocks'][3][2]['out_channels'] = 120
cfg['blocks'][3][2]['expanded_channels'] = 376
cfg['blocks'][3][2]['kernel_size'] = (3,3,3)
cfg['blocks'][3][2]['stride'] = (1,1,1)
cfg['blocks'][3][2]['padding'] = (1,1,1)
cfg['blocks'][3][2]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][3], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][3]['input_channels'] = 120
cfg['blocks'][3][3]['out_channels'] = 120
cfg['blocks'][3][3]['expanded_channels'] = 376
cfg['blocks'][3][3]['kernel_size'] = (3,3,3)
cfg['blocks'][3][3]['stride'] = (1,1,1)
cfg['blocks'][3][3]['padding'] = (1,1,1)
cfg['blocks'][3][3]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][4], 120, 120, 224, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][3][4]['input_channels'] = 120
cfg['blocks'][3][4]['out_channels'] = 120
cfg['blocks'][3][4]['expanded_channels'] = 224
cfg['blocks'][3][4]['kernel_size'] = (1,5,5)
cfg['blocks'][3][4]['stride'] = (1,1,1)
cfg['blocks'][3][4]['padding'] = (0,2,2)
cfg['blocks'][3][4]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][5], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][5]['input_channels'] = 120
cfg['blocks'][3][5]['out_channels'] = 120
cfg['blocks'][3][5]['expanded_channels'] = 376
cfg['blocks'][3][5]['kernel_size'] = (3,3,3)
cfg['blocks'][3][5]['stride'] = (1,1,1)
cfg['blocks'][3][5]['padding'] = (1,1,1)
cfg['blocks'][3][5]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][6], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][6]['input_channels'] = 120
cfg['blocks'][3][6]['out_channels'] = 120
cfg['blocks'][3][6]['expanded_channels'] = 376
cfg['blocks'][3][6]['kernel_size'] = (3,3,3)
cfg['blocks'][3][6]['stride'] = (1,1,1)
cfg['blocks'][3][6]['padding'] = (1,1,1)
cfg['blocks'][3][6]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][7], 120, 120, 224, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][7]['input_channels'] = 120
cfg['blocks'][3][7]['out_channels'] = 120
cfg['blocks'][3][7]['expanded_channels'] = 224
cfg['blocks'][3][7]['kernel_size'] = (3,3,3)
cfg['blocks'][3][7]['stride'] = (1,1,1)
cfg['blocks'][3][7]['padding'] = (1,1,1)
cfg['blocks'][3][7]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][8], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][8]['input_channels'] = 120
cfg['blocks'][3][8]['out_channels'] = 120
cfg['blocks'][3][8]['expanded_channels'] = 376
cfg['blocks'][3][8]['kernel_size'] = (3,3,3)
cfg['blocks'][3][8]['stride'] = (1,1,1)
cfg['blocks'][3][8]['padding'] = (1,1,1)
cfg['blocks'][3][8]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][9], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][9]['input_channels'] = 120
cfg['blocks'][3][9]['out_channels'] = 120
cfg['blocks'][3][9]['expanded_channels'] = 376
cfg['blocks'][3][9]['kernel_size'] = (3,3,3)
cfg['blocks'][3][9]['stride'] = (1,1,1)
cfg['blocks'][3][9]['padding'] = (1,1,1)
cfg['blocks'][3][9]['padding_avg'] = (0,1,1)
# cfg['blocks'][3][10], 120, 120, 376, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][3][10]['input_channels'] = 120
cfg['blocks'][3][10]['out_channels'] = 120
cfg['blocks'][3][10]['expanded_channels'] = 376
cfg['blocks'][3][10]['kernel_size'] = (3,3,3)
cfg['blocks'][3][10]['stride'] = (1,1,1)
cfg['blocks'][3][10]['padding'] = (1,1,1)
cfg['blocks'][3][10]['padding_avg'] = (0,1,1)

# block 6
# cfg['blocks'][4][0], 120 , 224, 744, (5,3,3), (1,2,2), (2,1,1), (0,1,1)
cfg['blocks'][4][0]['input_channels'] = 120
cfg['blocks'][4][0]['out_channels'] = 224
cfg['blocks'][4][0]['expanded_channels'] = 744
cfg['blocks'][4][0]['kernel_size'] = (5,3,3)
cfg['blocks'][4][0]['stride'] = (1,2,2)
cfg['blocks'][4][0]['padding'] = (2,1,1)
cfg['blocks'][4][0]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][1], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][4][1]['input_channels'] = 224
cfg['blocks'][4][1]['out_channels'] = 224
cfg['blocks'][4][1]['expanded_channels'] = 744
cfg['blocks'][4][1]['kernel_size'] = (3,3,3)
cfg['blocks'][4][1]['stride'] = (1,1,1)
cfg['blocks'][4][1]['padding'] = (1,1,1)
cfg['blocks'][4][1]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][2], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][2]['input_channels'] = 224
cfg['blocks'][4][2]['out_channels'] = 224
cfg['blocks'][4][2]['expanded_channels'] = 600
cfg['blocks'][4][2]['kernel_size'] = (1,5,5)
cfg['blocks'][4][2]['stride'] = (1,1,1)
cfg['blocks'][4][2]['padding'] = (0,2,2)
cfg['blocks'][4][2]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][3], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][3]['input_channels'] = 224
cfg['blocks'][4][3]['out_channels'] = 224
cfg['blocks'][4][3]['expanded_channels'] = 600
cfg['blocks'][4][3]['kernel_size'] = (1,5,5)
cfg['blocks'][4][3]['stride'] = (1,1,1)
cfg['blocks'][4][3]['padding'] = (0,2,2)
cfg['blocks'][4][3]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][4], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][4]['input_channels'] = 224
cfg['blocks'][4][4]['out_channels'] = 224
cfg['blocks'][4][4]['expanded_channels'] = 744
cfg['blocks'][4][4]['kernel_size'] = (1,5,5)
cfg['blocks'][4][4]['stride'] = (1,1,1)
cfg['blocks'][4][4]['padding'] = (0,2,2)
cfg['blocks'][4][4]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][5], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][5]['input_channels'] = 224
cfg['blocks'][4][5]['out_channels'] = 224
cfg['blocks'][4][5]['expanded_channels'] = 744
cfg['blocks'][4][5]['kernel_size'] = (1,5,5)
cfg['blocks'][4][5]['stride'] = (1,1,1)
cfg['blocks'][4][5]['padding'] = (0,2,2)
cfg['blocks'][4][5]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][6], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][4][6]['input_channels'] = 224
cfg['blocks'][4][6]['out_channels'] = 224
cfg['blocks'][4][6]['expanded_channels'] = 744
cfg['blocks'][4][6]['kernel_size'] = (3,3,3)
cfg['blocks'][4][6]['stride'] = (1,1,1)
cfg['blocks'][4][6]['padding'] = (0,2,2)
cfg['blocks'][4][6]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][7], 224, 224, 896, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][7]['input_channels'] = 224
cfg['blocks'][4][7]['out_channels'] = 224
cfg['blocks'][4][7]['expanded_channels'] = 896
cfg['blocks'][4][7]['kernel_size'] = (1,5,5)
cfg['blocks'][4][7]['stride'] = (1,1,1)
cfg['blocks'][4][7]['padding'] = (0,2,2)
cfg['blocks'][4][7]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][8], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][8]['input_channels'] = 224
cfg['blocks'][4][8]['out_channels'] = 224
cfg['blocks'][4][8]['expanded_channels'] = 600
cfg['blocks'][4][8]['kernel_size'] = (1,5,5)
cfg['blocks'][4][8]['stride'] = (1,1,1)
cfg['blocks'][4][8]['padding'] = (0,2,2)
cfg['blocks'][4][8]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][9], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][9]['input_channels'] = 224
cfg['blocks'][4][9]['out_channels'] = 224
cfg['blocks'][4][9]['expanded_channels'] = 600
cfg['blocks'][4][9]['kernel_size'] = (1,5,5)
cfg['blocks'][4][9]['stride'] = (1,1,1)
cfg['blocks'][4][9]['padding'] = (0,2,2)
cfg['blocks'][4][9]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][10], 224, 224, 896, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][10]['input_channels'] = 224
cfg['blocks'][4][10]['out_channels'] = 224
cfg['blocks'][4][10]['expanded_channels'] = 896
cfg['blocks'][4][10]['kernel_size'] = (1,5,5)
cfg['blocks'][4][10]['stride'] = (1,1,1)
cfg['blocks'][4][10]['padding'] = (0,2,2)
cfg['blocks'][4][10]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][11], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][11]['input_channels'] = 224
cfg['blocks'][4][11]['out_channels'] = 224
cfg['blocks'][4][11]['expanded_channels'] = 744
cfg['blocks'][4][11]['kernel_size'] = (1,5,5)
cfg['blocks'][4][11]['stride'] = (1,1,1)
cfg['blocks'][4][11]['padding'] = (0,2,2)
cfg['blocks'][4][11]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][12], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][4][12]['input_channels'] = 224
cfg['blocks'][4][12]['out_channels'] = 224
cfg['blocks'][4][12]['expanded_channels'] = 744
cfg['blocks'][4][12]['kernel_size'] = (3,3,3)
cfg['blocks'][4][12]['stride'] = (1,1,1)
cfg['blocks'][4][12]['padding'] = (1,1,1)
cfg['blocks'][4][12]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][13], 224, 224, 896, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][13]['input_channels'] = 224
cfg['blocks'][4][13]['out_channels'] = 224
cfg['blocks'][4][13]['expanded_channels'] = 896
cfg['blocks'][4][13]['kernel_size'] = (1,5,5)
cfg['blocks'][4][13]['stride'] = (1,1,1)
cfg['blocks'][4][13]['padding'] = (0,2,2)
cfg['blocks'][4][13]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][14], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][14]['input_channels'] = 224
cfg['blocks'][4][14]['out_channels'] = 224
cfg['blocks'][4][14]['expanded_channels'] = 600
cfg['blocks'][4][14]['kernel_size'] = (1,5,5)
cfg['blocks'][4][14]['stride'] = (1,1,1)
cfg['blocks'][4][14]['padding'] = (0,2,2)
cfg['blocks'][4][14]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][15], 224, 224, 600, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][15]['input_channels'] = 224
cfg['blocks'][4][15]['out_channels'] = 224
cfg['blocks'][4][15]['expanded_channels'] = 600
cfg['blocks'][4][15]['kernel_size'] = (1,5,5)
cfg['blocks'][4][15]['stride'] = (1,1,1)
cfg['blocks'][4][15]['padding'] = (0,2,2)
cfg['blocks'][4][15]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][16], 224, 224, 744, (1,5,5), (1,1,1), (0,2,2), (0,1,1)
cfg['blocks'][4][16]['input_channels'] = 224
cfg['blocks'][4][16]['out_channels'] = 224
cfg['blocks'][4][16]['expanded_channels'] = 744
cfg['blocks'][4][16]['kernel_size'] = (1,5,5)
cfg['blocks'][4][16]['stride'] = (1,1,1)
cfg['blocks'][4][16]['padding'] = (0,2,2)
cfg['blocks'][4][16]['padding_avg'] = (0,1,1)
# cfg['blocks'][4][17], 224, 224, 744, (3,3,3), (1,1,1), (1,1,1), (0,1,1)
cfg['blocks'][4][17]['input_channels'] = 224
cfg['blocks'][4][17]['out_channels'] = 224
cfg['blocks'][4][17]['expanded_channels'] = 744
cfg['blocks'][4][17]['kernel_size'] = (3,3,3)
cfg['blocks'][4][17]['stride'] = (1,1,1)
cfg['blocks'][4][17]['padding'] = (1,1,1)
cfg['blocks'][4][17]['padding_avg'] = (0,1,1)


# cfg['conv7'], 224,992,(1,1,1),(1,1,1),(0,0,0)
cfg['conv7'] = dict()
cfg['conv7']['input_channels'] = 224
cfg['conv7']['out_channels'] = 992
cfg['conv7']['kernel_size'] = (1,1,1)
cfg['conv7']['stride'] = (1,1,1)
cfg['conv7']['padding'] = (0,0,0)


cfg['dense9'] = dict()
cfg['dense9']['hidden_dim'] = 2048



model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MoViNet',
        cfg=cfg,
        causal=False,
        conv_type="3d",
        tf_like=True
    ),
    cls_head=dict(
        type='MoViNetHead',
        num_classes=600,
        dropout_ratio=0.5
    ),
    train_cfg=None,
    test_cfg=dict(maximize_clips='score')
)