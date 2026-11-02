# 2D Baseball Ball Tracking Prototype — 日本語README

## 含まれるファイル
- `baseball_robust_2d_tracking.mp4`: 元動画に追跡結果を重ねた動画
- `baseball_robust_2d_tracking.csv`: フレーム単位の座標・状態・信頼度
- `baseballcv_robust_2d_tracker.py`: 再実行できる追跡コード
- `experiment_report_ja.md`: 実験条件・評価・限界

## 動画の色分け
- **水色リング**: リリース位置候補（この動画では frame 11 / 約0.367 秒）
- **赤い四角**: 検出器が選んだボール中心。重複フレームでは直前検出を複写
- **緑線**: ボールが視認できる区間の追跡軌跡
- **オレンジ破線・丸**: オクルージョン中の外挿位置
- **オレンジ十字**: ミット付近とした推定受球領域（実際の捕球の検出ではない）

## 実行方法

OpenCV フォールバックで動かす場合：

```bash
pip install opencv-python numpy
python baseballcv_robust_2d_tracker.py paligemma2_output.mp4 \
  --output baseball_robust_2d_tracking.mp4 \
  --csv baseball_robust_2d_tracking.csv --backend opencv
```

BaseballCV の YOLO `ball_tracking` 重みを利用できる環境では、次を試します。

```bash
pip install baseballcv ultralytics
python baseballcv_robust_2d_tracker.py paligemma2_output.mp4 \
  --output baseball_robust_2d_tracking_yolo.mp4 \
  --csv baseball_robust_2d_tracking_yolo.csv --backend auto
```

`auto` は BaseballCV / Ultralytics / 重みをロードできれば YOLO を使い、ロードに失敗すると OpenCV フォールバックに切り替えます。`--backend baseballcv-yolo` を指定すると YOLO が使えない場合はエラーで停止します。

## 重要な注意
今回生成した結果は、**学習済みニューラルネットをこの実行環境で実際にロードして得た結果ではありません**。この環境には `baseballcv`、`ultralytics`、`transformers` がなく、GitHub への接続も DNS エラーで失敗し、GPUも利用できませんでした。そのため、実際に実行できたのは輝度・彩度による球候補抽出、方向制約、近接候補統合、フレーム間の連続性を用いる OpenCV ベースラインです。

また、オレンジ区間は捕球位置の教師ラベルや実際の検出ではなく、視認可能な最後の位置からの短い外挿です。単眼画像から3D軌道を復元した結果ではありません。ネット、照明、ボール色、背景、カメラ位置が変わる場合は、同じ閾値のまま使う保証はありません。
