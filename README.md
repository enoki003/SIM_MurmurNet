# SLM Emergent AI - 使用方法ガイド

## 概要

SLM Emergent AIは、Small Language Model (SLM, ≤ 1B params)を複数体協調させ、Boidsアルゴリズムに触発されたプロンプトエンジニアリングとブラックボード共有メモリを活用し、中央制御なしの創発知能を実証するシステムです。エージェントの行動は `BoidsPromptEnhancer` を通じて誘導され、LLMへの入力プロンプトが調整されます。

このプロジェクトは設計書に基づいて実装されており、gemma3:1bモデルを使用して複数のエージェントが協調的に動作します。

## システム要件

- Python 3.8以上
- CPU環境（GPUは不要）
- メモリ: 最低8GB（推奨16GB以上）
- ディスク容量: 最低5GB

## インストール方法

1. リポジトリをクローンまたは解凍します
```bash
unzip slm_emergent_ai.zip
cd slm_emergent_ai
```

2. 必要なパッケージをインストールします
```bash
pip install -r requirements.txt
```

3. モデルをダウンロードします
```bash
./scripts/dl_model.sh gemma3:1b
```

4. （オプション）モデルを量子化します
```bash
./scripts/quantize.sh gemma3:1b q4
```

## テスト (Testing)

本プロジェクトにはユニットテストが含まれています。テストは、コアエージェントロジック、メトリック計算、エンベディングサービス、Boidsプロンプトエンハンスメントなどを対象としています。

すべてのユニットテストを実行するには、リポジトリのルートディレクトリから以下のコマンドを実行してください:

```bash
./run_all_tests.py
```

## 使用方法

### 基本的な実行方法

```bash
python -m slm_emergent_ai.run_sim -c slm_emergent_ai/configs/base.yml
```

### コマンドラインオプション

- `-c, --config`: 設定ファイルのパス
- `-m, --multi`: 複数の設定を並列実行
- `--agent.n`: エージェント数
- `--model.path`: モデルパス
- `--runtime.threads`: スレッド数
- `--ui`: UIの有効化

例:
```bash
python -m slm_emergent_ai.run_sim -c slm_emergent_ai/configs/base.yml agent.n=5 model.path=models/gemma3-1b-q4 runtime.threads=8 ui=true
```

### UIダッシュボード

UIが有効な場合、以下のURLでダッシュボードにアクセスできます:
```
http://localhost:7860
```

ダッシュボードでは以下の機能が利用可能です:
- リアルタイムメトリクス（エントロピー、VDI、FCR、速度）の表示
- λパラメータの監視
- BlackBoardメッセージの表示
- プロンプトの直接注入

## プロジェクト構成

```
slm_emergent_ai/
├── data/               # seed prompts, fake‑fact corpus
├── docker/             # Dockerfile + compose for GPU / CPU
├── notebooks/          # Jupyter demo + exploratory analysis
├── scripts/            # bash helpers (download, convert‑gguf)
│   ├── dl_model.sh
│   └── quantize.sh
├── run_all_tests.py    # Add this line - Test execution script
└── slm_emergent_ai/
    ├── __init__.py
    ├── run_sim.py
    ├── configs/
    │   ├── base.yml
    │   ├── grid.yml  
    │   └── prod.yml 
    ├── models/
    │   ├── loader.py
    │   └── quantizer.py
    ├── agents/
    │   ├── core.py
    │   ├── rag.py
    │   ├── memory_agent.py
    │   └── boids_enhancer.py # Add this line - Boids prompt enhancement logic
    ├── boids/  # This directory might be less central now, or removed if all logic moved
    │   ├── rules.py
    │   └── processor.py
    ├── memory/
    │   ├── blackboard.py
    │   ├── db_sqlite.py 
    │   └── db_redis.py 
    ├── services/           # Add this section - Centralized services
    │   └── embedding_service.py
    ├── eval/
    │   ├── metrics.py
    │   ├── plot.py     
    │   └── benchmark.py 
    ├── controller/
    │   ├── meta.py
    │   └── pid.py
    ├── ui/
    │   ├── dashboard.py
    │   └── schema.py   
    └── tests/
        ├── test_boids.py 
        ├── test_blackboard.py 
        ├── test_metrics.py
        ├── test_embedding_service.py # Add this line
        └── test_boids_enhancer.py  # Add this line
        └── perf/
            └── bench_10agents.py 
```

## 主要コンポーネント

1. **SLMAgent**: Small Language Modelエージェント。`BoidsPromptEnhancer` から提供されるガイダンスに基づきプロンプトを調整して応答します。
2. **BlackBoard**: エージェント間の共有メモリ。`EmbeddingService` を利用してメッセージのベクトル表現を管理することがあります。
3. **MetaController**: λパラメータの自動調整。
4. **BoidsPromptEnhancer**: Boidsアルゴリズム（整列、結合、分離）の原則に基づき、エージェントのプロンプトを強化して協調行動を促進します。
5. **EmbeddingService**: テキストデータのベクトル埋め込みを生成する中央集権的なサービス。
6. **RAG**: 長期記憶のためのRetrieval Augmented Generation。`EmbeddingService` を利用します。
7. **Metrics**: エントロピー、VDI、FCR、速度などの評価指標。
8. **UI Dashboard**: リアルタイムモニタリングとコントロール。

## 注意事項

- 現在の実装はCPU環境に最適化されています
- 大規模な実験（30体以上）には十分なメモリが必要です
- 一部のモジュールは今後の拡張のために準備されています

## トラブルシューティング

1. **メモリ不足エラー**
   - `--agent.n`を減らす
   - モデルをより高度に量子化する（q4）
   - `OMP_NUM_THREADS`環境変数を設定する

2. **速度が遅い**
   - `runtime.threads`を増やす
   - より軽量なモデルを使用する

3. **UIが応答しない**
   - ブラウザをリフレッシュする
   - メトリクスの更新間隔を長くする

## 今後の拡張予定

- 分散環境でのスケーリング
- より多様なBoidsルールの実装
- ベンチマークスイートの追加
- GPU対応の強化

## 参考文献

1. Reynolds, C. 1987. *Flocks, herds and schools: A distributed behavioral model.* SIGGRAPH.
2. Mu & Viswanath. 2018. *All‑but‑the‑Top: Simple and effective postprocessing for word representations.* ACL.
3. Li et al. 2023. *CAMEL: Communicative Agents for "Mind" Exploration.* arXiv.
4. Park et al. 2023. *Generative Agents: Interactive Simulacra of Human Behavior.* arXiv 2304.03442.
5. Deutsch, D. 2022. *Constructor Theory of Information.* OUP.
6. Holland, J. H. 1992. *Adaptation in Natural and Artificial Systems.* MIT Press.
