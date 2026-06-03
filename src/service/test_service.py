'''
用法：python src/service/test_service.py [--url http://localhost:7788]
'''

import argparse, json, requests

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://localhost:7788')
args = parser.parse_args()


CASE_INDEX = 0

# 16-hand 变种：每人 16 张，牌池 48 张（3-K 各 4，A=3，2=1）
payload = {
    'game_id':      'test',
    'player_id':    'player',
    'player_index': 0, 
    'hand_cards':   ['9', '3334459TAA'],
    'actions':      ['3456789TJ', '', '5', 'K', 'A', '2', '', '6677', 'QQKK', ''],
    'first_player': 0
}
    

print('Request:', json.dumps(payload, ensure_ascii=False))
resp = requests.post(f'{args.url}/suggestion', json=payload, timeout=10)
print('Response:', json.dumps(resp.json(), ensure_ascii=False))
