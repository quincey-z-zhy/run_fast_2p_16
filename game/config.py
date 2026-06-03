# 牌面与牌值的对应关系
CARD_TO_INT = {
    '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 15
}
CARD_TO_CHAR = {v: k for k, v in CARD_TO_INT.items()}

# 所有牌值（升序）
FACES = sorted(CARD_TO_INT.values())  # [3, 4, ..., 15]

# 可以成顺（顺子）的牌值：3-A（单张顺子只需各 1 张，2 除外）
MIN_FACE_STRAIGHT = CARD_TO_INT['3']
MAX_FACE_STRAIGHT = CARD_TO_INT['A']
STRAIGHT_FACES = list(range(MIN_FACE_STRAIGHT, MAX_FACE_STRAIGHT + 1))  # [3..14]

# 可以成对、三条的牌值：3-A（均有 ≥3 张）
POCKET_TRIPLE_FACES = list(range(3, 15))  # [3..14]

# 可以成炸弹或四带的牌值：3-K（均有 4 张）
QUAD_FACES = list(range(3, 14))  # [3..13]

# 牌池各牌值的张数
DECK_COUNTS = {face: 4 for face in range(3, 14)}  # 3-K: 各 4 张
DECK_COUNTS[14] = 3   # A: 3 张
DECK_COUNTS[15] = 1   # 2: 1 张
# 合计 48 张

# 每人初始手牌数
HAND_SIZE = 16
MAX_AMOUNT = HAND_SIZE

# 顺子长度范围（张数）
MIN_STRAIGHT_LEN = 5
MAX_STRAIGHT_LEN = len(STRAIGHT_FACES)  # 最长可达 3-A 共 12 张

# 连对长度范围（对数）
MIN_S_POCKET_LEN = 2
MAX_S_POCKET_LEN = 7

# 飞机长度范围（三条节数）
MIN_S_TRIPS_LEN = 2
MAX_S_TRIPS_LEN = 4

# 动作牌型的整型编码
FORMATIONS = {
    'none':     0,
    'single':   1,
    'pocket':   2,
    's_pocket': 3,
    'trips':    8,
    's_trips':  9,
    'quads':    12,
    'straight': 13,
    'bomb':     14,
}

# 牌型的中文名
FORMATION_NAMES = {
    FORMATIONS['none']:     '不出',
    FORMATIONS['single']:   '单牌',
    FORMATIONS['pocket']:   '对子',
    FORMATIONS['s_pocket']: '连对',
    FORMATIONS['trips']:    '三条',
    FORMATIONS['s_trips']:  '飞机',
    FORMATIONS['quads']:    '四带',
    FORMATIONS['straight']: '顺子',
    FORMATIONS['bomb']:     '炸弹',
}

# 得分参数
CARD_SCORE  = 1    # 基础分（赢家出完的 16 张 × 1）
BOMB_BONUS  = 10   # 每个炸弹的即时加分
SPRING_SCORE = 32  # 春天奖励（= 2 × HAND_SIZE，16-hand 变种）
