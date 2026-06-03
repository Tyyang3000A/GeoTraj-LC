import argparse
import json
import os

import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

from geotraj_lc.anchor_correction.model import OffsetRegressor


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_detections(det_path, seq_name):
    file_path = os.path.join(det_path, f"{seq_name}.txt")
    if not os.path.exists(file_path):
        return {}

    dets = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                frame_id = int(parts[0])
                x1 = float(parts[2])
                y1 = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                if frame_id not in dets:
                    dets[frame_id] = []
                dets[frame_id].append({"x1": x1, "y1": y1, "w": w, "h": h, "parts": parts})
            except ValueError:
                continue
    return dets


def correct_detections(config_path, model_path=None, output_dir=None, target_seqs=None):
    config = load_config(config_path)

    det_path = config.get("save_config", {}).get("det_path", "")
    video_source = config.get("source", {}).get("video_source", "")
    channels = config.get("source", {}).get("channels", {})

    if target_seqs:
        print(f"Target sequences specified: {target_seqs}")
        channels = {k: v for k, v in channels.items() if k in target_seqs}
        if not channels:
            print(f"Error: None of the specified sequences {target_seqs} were found in config.")
            return 1

    if model_path is None:
        model_path = config.get("detector_config", {}).get("refinement_model_path")

    if output_dir is None:
        output_dir = config.get("save_config", {}).get("det_corrected_path")

    if not model_path:
        print("Error: refinement_model_path not found in config['detector_config']")
        return 1
    if not output_dir:
        print("Warning: det_corrected_path not found in config['save_config'], using default.")
        output_dir = "corrected_dets"

    print(f"Config: {config_path}")
    print(f"Video: {video_source}")
    print(f"Detections: {det_path}")
    print(f"Model Path: {model_path}")
    print(f"Output: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = OffsetRegressor()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Error: Model not found at {model_path}")
        return 1

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    BATCH_SIZE = 64

    for seq_name, crop_rect in channels.items():
        print(f"\nProcessing sequence: {seq_name}")

        seq_dets = get_detections(det_path, seq_name)
        if not seq_dets:
            print(f"Warning: No detections found for {seq_name}")
            continue

        corrected_lines = []

        batch_tensors = []
        pending_entries = []

        def flush_batch():
            nonlocal batch_tensors, pending_entries, corrected_lines
            if not batch_tensors and not pending_entries:
                return

            pred_coords_batch = None
            if batch_tensors:
                input_batch = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    pred_coords, _ = model(input_batch)
                    pred_coords_batch = pred_coords.cpu().numpy()

            pred_idx = 0
            for entry in pending_entries:
                if not entry["needs_inference"]:
                    corrected_lines.append(entry["line"])
                    continue

                if pred_coords_batch is None:
                    continue

                lx_norm, ly_norm, rx_norm, ry_norm = pred_coords_batch[pred_idx]
                pred_idx += 1

                crop_x1 = entry["crop_x1"]
                crop_y1 = entry["crop_y1"]
                patch_w = entry["patch_w"]
                patch_h = entry["patch_h"]
                x1, y1, w, h = entry["orig_coords"]
                parts = entry["parts"]
                ch_w = entry["ch_w"]

                pred_lx = crop_x1 + lx_norm * patch_w
                pred_rx = crop_x1 + rx_norm * patch_w
                pred_ly = crop_y1 + ly_norm * patch_h
                pred_ry = crop_y1 + ry_norm * patch_h

                new_x1 = pred_lx
                new_w = pred_rx - pred_lx

                if new_w <= 0:
                    new_x1, new_y1, new_w, new_h = x1, y1, w, h
                else:
                    new_y1 = y1
                    new_y2 = max(pred_ly, pred_ry)
                    new_h = new_y2 - new_y1

                hw_ratio = new_h / new_w if new_w > 0 else 1.0
                if hw_ratio >= 5 and (x1 < 20 or x1 + w > (ch_w - 20)):
                    continue

                parts[2] = f"{new_x1:.0f}"
                parts[3] = f"{new_y1:.0f}"
                parts[4] = f"{new_w:.0f}"
                parts[5] = f"{new_h:.0f}"
                corrected_lines.append(",".join(parts) + "\n")

            batch_tensors.clear()
            pending_entries.clear()

        video_file_path = os.path.join(video_source, f"{seq_name}.mp4")
        if not os.path.exists(video_file_path):
            video_file_path = os.path.join(video_source, f"{seq_name}.avi")

        if not os.path.exists(video_file_path) and os.path.isfile(video_source):
            video_file_path = video_source

        cap = cv2.VideoCapture(video_file_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_file_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = 100000

        pbar = tqdm(total=total_frames, desc=f"Processing {seq_name}")

        frame_idx = 0
        while True:
            if frame_idx >= total_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            pbar.update(1)

            if frame_idx not in seq_dets:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_img, w_img, _ = frame.shape

            cx1, cy1, cx2, cy2 = crop_rect
            cx1, cy1 = max(0, cx1), max(0, cy1)
            cx2, cy2 = min(w_img, cx2), min(h_img, cy2)

            channel_img = frame_rgb[cy1:cy2, cx1:cx2]
            ch_h, ch_w, _ = channel_img.shape

            if ch_h == 0 or ch_w == 0:
                continue

            current_dets = seq_dets[frame_idx]

            for det in current_dets:
                x1, y1, w, h = det["x1"], det["y1"], det["w"], det["h"]
                parts = det["parts"]

                pad_w = w * 0.1
                pad_h = h * 0.1

                crop_x1 = max(0, int(x1 - pad_w))
                crop_y1 = max(0, int(y1 - pad_h))
                crop_x2 = min(ch_w, int(x1 + w + pad_w))
                crop_y2 = min(ch_h, int(y1 + h + pad_h))

                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    pending_entries.append({
                        "needs_inference": False,
                        "line": ",".join(parts) + "\n",
                    })
                    continue

                patch = channel_img[crop_y1:crop_y2, crop_x1:crop_x2]
                patch_w = crop_x2 - crop_x1
                patch_h = crop_y2 - crop_y1

                batch_tensors.append(transform(patch))
                pending_entries.append({
                    "needs_inference": True,
                    "parts": parts,
                    "crop_x1": crop_x1,
                    "crop_y1": crop_y1,
                    "patch_w": patch_w,
                    "patch_h": patch_h,
                    "orig_coords": (x1, y1, w, h),
                    "ch_w": ch_w,
                })

                if len(batch_tensors) >= BATCH_SIZE:
                    flush_batch()

        flush_batch()
        cap.release()
        pbar.close()

        if corrected_lines:
            out_file = os.path.join(output_dir, f"{seq_name}.txt")
            with open(out_file, "w") as f:
                f.writelines(corrected_lines)
            print(f"Saved corrected detections to {out_file}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Correct detections using trained OffsetRegressor")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--model", type=str, default=None, help="Path to trained regressor weights")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=None, help="Directory to save corrected detection files")
    parser.add_argument("--seqs", nargs="+", default=None, help="Space-separated sequence names to process")
    args = parser.parse_args()

    return correct_detections(
        config_path=args.config,
        model_path=args.model,
        output_dir=args.output_dir,
        target_seqs=args.seqs,
    )


if __name__ == "__main__":
    raise SystemExit(main())

