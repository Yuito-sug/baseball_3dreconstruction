import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import supervision as sv

from PIL import Image
from peft import PeftModel
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from baseballcv.functions import LoadTools


# ==================================================
# 1. パス・設定
# ==================================================

VIDEO_PATH = r"input_video/pitch_sample.mp4"

OUTPUT_VIDEO = r"output_video/paligemma2/paligemma2_output.mp4"
OUTPUT_CSV = r"output_video/paligemma2/paligemma2_trajectory.csv"

MODEL_NAME = "paligemma2_ball_tracking"
MODEL_TYPE = "PALIGEMMA2"

# BaseballCVの物体検出用プロンプト
TEXT_PROMPT = "baseball"

# Noneなら動画全体を処理する
# 動作確認だけなら 10 などに変更する
MAX_FRAMES = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# 2. モデル読み込み
# ==================================================
def load_paligemma2():

    print("Loading BaseballCV PaliGemma2 model...")

    project_dir = Path(
        r"/home/yuito/myresearch/baseball3d/3d_ball_tracking/2d_ball_detection"
    )
    
    adapter_path = (
        project_dir
        / "model_weights"
        / "paligemma2_ball_tracking"
    )

    adapter_config_path = adapter_path / "adapter_config.json"

    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"adapter_config.json not found: {adapter_config_path}"
        )

    print(f"Adapter directory: {adapter_path}")

    # LoRAアダプターの設定を読み込む
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_id = adapter_config.get(
        "base_model_name_or_path"
    )

    if not base_model_id:
        raise ValueError(
            "base_model_name_or_path is missing from adapter_config.json"
        )

    print(f"Base model: {base_model_id}")
    print(f"Device: {DEVICE}")

    # ベースモデルをロード
    base_model = (
        PaliGemmaForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
    )

    # ローカルのLoRAアダプターを適用
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )

    model.to(DEVICE)
    model.eval()

    # Processorをロード
    processor = PaliGemmaProcessor.from_pretrained(
        str(adapter_path)
    )

    print("PaliGemma2 model loaded successfully.")

    return model, processor

# ==================================================
# 3. 1フレームからボールを検出
# ==================================================

def detect_ball(model, processor, frame):

    height, width = frame.shape[:2]

    # OpenCVのBGR画像をRGB画像に変換
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    # BaseballCVのPaliGemma2実装に合わせた入力形式
    prompt = f"<image>{TEXT_PROMPT}"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[-1]

    # 推論
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=3,
        )

    # 入力プロンプト部分を除いて生成結果を取得
    generated_ids = output_ids[0][input_length:]

    generated_text = processor.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # PaliGemma形式の物体検出結果を座標に変換
    detections = sv.Detections.from_lmm(
        lmm="paligemma",
        result=generated_text,
        resolution_wh=(width, height),
    )

    boxes = np.asarray(
        detections.xyxy,
        dtype=float,
    ).reshape(-1, 4)

    if len(boxes) == 0:
        return None, None, None, generated_text

    # 異常な座標を除外
    valid_indices = [
        i
        for i, box in enumerate(boxes)
        if (
            np.all(np.isfinite(box))
            and box[2] > box[0]
            and box[3] > box[1]
        )
    ]

    if not valid_indices:
        return None, None, None, generated_text

    # クラス名がある場合、野球ボールの候補を優先
    labels = detections.data.get("class_name")

    if labels is not None and len(labels) == len(boxes):

        ball_indices = [
            i for i in valid_indices
            if (
                "baseball" in str(labels[i]).lower()
                or str(labels[i]).lower() == "ball"
            )
        ]

        # プロンプトに対する結果にクラス名が一致しない場合は、
        # 有効な候補を残しておく
        candidates = ball_indices or valid_indices

    else:
        candidates = valid_indices

    # 各候補の中心座標
    centers = {
        i: (
            (boxes[i, 0] + boxes[i, 2]) / 2,
            (boxes[i, 1] + boxes[i, 3]) / 2,
        )
        for i in candidates
    }

    # 最初の候補を選択。
    # 複数候補があるときの選択はメイン処理で行う。
    selected_index = candidates[0]

    selected_box = boxes[selected_index]
    center_x, center_y = centers[selected_index]

    return (
        selected_box,
        (center_x, center_y),
        selected_index,
        generated_text,
    )


# ==================================================
# 4. 動画処理
# ==================================================

def main():

    input_path = Path(VIDEO_PATH)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input video not found: {input_path.resolve()}"
        )

    # 出力先を作成
    output_video_path = Path(OUTPUT_VIDEO)
    output_csv_path = Path(OUTPUT_CSV)

    output_video_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    output_csv_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    # モデルをロード
    model, processor = load_paligemma2()

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    frame_limit = (
        total_frames
        if MAX_FRAMES is None
        else min(total_frames, MAX_FRAMES)
    )

    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.3f}")
    print(f"Frames to process: {frame_limit}")

    # MP4出力
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video: {OUTPUT_VIDEO}"
        )

    frame_index = 0
    detection_count = 0

    # 過去の検出中心座標
    trajectory = []
    previous_center = None

    previous_detection_frame = None

    try:
        with open(
            output_csv_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow([
                "frame_index",
                "timestamp_sec",
                "detected",
                "x_center",
                "y_center",
                "x1",
                "y1",
                "x2",
                "y2",
                "generated_text",
            ])

            while frame_index < frame_limit:

                ret, frame = cap.read()

                if not ret:
                    break

                timestamp_sec = frame_index / fps

                (
                    box,
                    center,
                    selected_index,
                    generated_text,
                ) = detect_ball(model, processor, frame)

                if frame_index == 0:
                    print(
                        "First-frame model output:",
                        repr(generated_text),
                    )

                if box is not None:

                    # フレーム間で複数候補を選択する場合、
                    # 直近の検出中心に最も近い候補が望ましい。
                    # detect_ballでは現在の候補1つを返すため、
                    # ここでは返された中心を使用する。
                    center_x, center_y = center

                    x1, y1, x2, y2 = box

                    # 画像の範囲内に座標を収める
                    x1 = int(np.clip(round(x1), 0, width - 1))
                    y1 = int(np.clip(round(y1), 0, height - 1))
                    x2 = int(np.clip(round(x2), 0, width - 1))
                    y2 = int(np.clip(round(y2), 0, height - 1))

                    center_x = int(np.clip(
                        round(center_x), 0, width - 1
                    ))
                    center_y = int(np.clip(
                        round(center_y), 0, height - 1
                    ))

                    # CSVへ記録
                    csv_writer.writerow([
                        frame_index,
                        round(timestamp_sec, 6),
                        True,
                        center_x,
                        center_y,
                        x1,
                        y1,
                        x2,
                        y2,
                        generated_text,
                    ])

                    # バウンディングボックス
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    # ボール中心
                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        4,
                        (0, 0, 255),
                        -1,
                    )

                    cv2.putText(
                        frame,
                        "Baseball",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    # 2D軌跡
                    trajectory.append((center_x, center_y))

                    # 検出点をつなぐ
                    if len(trajectory) >= 2:

                        points = np.asarray(
                            trajectory[-60:],
                            dtype=np.int32,
                        ).reshape((-1, 1, 2))

                        cv2.polylines(
                            frame,
                            [points],
                            isClosed=False,
                            color=(0, 255, 255),
                            thickness=2,
                        )

                    previous_center = (center_x, center_y)
                    previous_detection_frame = frame_index
                    detection_count += 1

                else:
                    # 未検出フレームもCSVに残す
                    csv_writer.writerow([
                        frame_index,
                        round(timestamp_sec, 6),
                        False,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        generated_text,
                    ])

                cv2.putText(
                    frame,
                    f"Frame: {frame_index}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                writer.write(frame)

                frame_index += 1

                if frame_index % 10 == 0:
                    print(
                        f"Processed {frame_index}/{frame_limit} "
                        f"| detected frames: {detection_count}"
                    )

    finally:
        cap.release()
        writer.release()

    print("\nProcessing completed.")
    print(f"Processed frames: {frame_index}")
    print(f"Detected frames: {detection_count}")
    print(f"Video: {output_video_path.resolve()}")
    print(f"CSV: {output_csv_path.resolve()}")


if __name__ == "__main__":
    main()