# TicTacToe-RL
- [詳しくはこちらの記事へ](https://qiita.com/ysk0832/items/f054e100775cc790d1cd)
- [Google Colab で学習済みエージェントと対戦できます](https://colab.research.google.com/drive/1AfgMy6YQmnakq0RQlCttw4pk0v-awD2f?usp=sharing)
- 三目並べを強化学習（Q 学習）で攻略するプログラム
- 盤面の大きさが 4x4 のときにも対応

## ローカル環境で試す
```bash
git clone https://github.com/yousukeayada/TicTacToe-RL.git
cd TicTacToe-RL

python demo.py --size=3

# 学習させる場合（エピソード数に注意）
python train.py --size=3 --alpha=0.1 --gamma=0.9
```
