README / 操作マニュアル
(コーディング途上であり、以下の記述は正確ではありません）
本リポジトリでは main_rl_temp.py に用意した主要フラグを使い、DQN学習→Expertデータ収集→BC学習→BCポリシー評価の一連パイプラインを実行できます。以下に各フラグの機能、実行例、想定出力サマリをまとめます。

フラグ一覧と実行例
--train_dqn DQN単独学習モードを起動し，data/dqn_model.pth を出力します。 実行例:

bash
python3 main_rl_temp.py --train_dqn --num_episodes 2000
想定出力:

[DQN Training Mode]

各エピソードごとの報酬表示（例 [DQN Ep0] reward=123.45）

Saved DQN model → data/dqn_model.pth

--collect_expert 学習済みDQNを用い，Expertトラジェクトリを data/expert_data.pkl に保存します。 実行例:

bash
python3 main_rl_temp.py --collect_expert \
  --num_expert_episodes 1000 \
  --expert_threshold 10000
想定出力:

[Collect Expert Mode]

各エピソードのステップ数・報酬ログ

Collected 50000 transitions → data/expert_data.pkl

--bc_train Expertデータを使ってBehavioral Cloningを学習し，data/bc_model.pth を出力します。 実行例:

bash
python3 main_rl_temp.py --bc_train \
  --bc_lr 1e-3 \
  --bc_epochs 50
想定出力:

[BC Training Mode]

各エポックの損失・精度ログ

Saved BC model → data/bc_model.pth

--eval_bc 学習済BCモデルを評価モードで実行し，平均報酬を計測します。 実行例:

bash
python3 main_rl_temp.py --eval_bc \
  --num_eval_episodes 100
想定出力:

=== BC Evaluation Mode ===

各エピソードの報酬表示（例 [Ep0] reward=45.67）

Average Reward over 100 episodes: 50.12

全モード一気通しコマンド
以下のワンライナーで DQN学習→Expert収集→BC学習→BC評価 を順次実行します。

bash
python3 main_rl_temp.py --train_dqn --num_episodes 1000 && \
python3 main_rl_temp.py --collect_expert \
  --num_expert_episodes 1000 --expert_threshold 10000 && \
python3 main_rl_temp.py --bc_train \
  --bc_lr 1e-3 --bc_epochs 50 && \
python3 main_rl_temp.py --eval_bc --num_eval_episodes 100
