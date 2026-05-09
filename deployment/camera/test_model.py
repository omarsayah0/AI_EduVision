import argparse
import os
import cv2
import json
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CONF_THRESH = 0.25   # lower threshold for testing (main.py uses 0.6)


def mode_info(model):
    print("\n===== MODEL INFO =====")
    print(f"Classes : {model.names}")
    print(f"Task    : {model.task}")
    model.info(verbose=True)


def mode_val(model, data_yaml):
    """Validate on a labeled YOLO dataset → prints mAP, precision, recall."""
    if not os.path.exists(data_yaml):
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        return

    print(f"\n===== VALIDATION on {data_yaml} =====")
    metrics = model.val(
        data=data_yaml,
        conf=CONF_THRESH,
        iou=0.5,
        verbose=True,
        plots=True,          # saves confusion matrix, PR curve, etc.
        save_json=True,
    )

    print("\n===== ACCURACY RESULTS =====")
    print(f"mAP50        : {metrics.box.map50:.4f}  ({metrics.box.map50*100:.1f}%)")
    print(f"mAP50-95     : {metrics.box.map:.4f}  ({metrics.box.map*100:.1f}%)")
    print(f"Precision    : {metrics.box.mp:.4f}  ({metrics.box.mp*100:.1f}%)")
    print(f"Recall       : {metrics.box.mr:.4f}  ({metrics.box.mr*100:.1f}%)")

    print("\nPer-class results:")
    for i, name in model.names.items():
        p  = metrics.box.p[i]  if i < len(metrics.box.p)  else 0
        r  = metrics.box.r[i]  if i < len(metrics.box.r)  else 0
        ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
        print(f"  {name:<20} P={p:.3f}  R={r:.3f}  AP50={ap:.3f}")

    print("\nPlots saved to: runs/detect/val*/")


def mode_images(model, source_dir):
    """Run inference on all images in a folder and print a confidence report."""
    source_dir = Path(source_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in source_dir.iterdir() if p.suffix.lower() in exts]

    if not images:
        print(f"[ERROR] No images found in {source_dir}")
        return

    print(f"\n===== INFERENCE on {len(images)} images =====\n")

    results_summary = []
    for img_path in sorted(images):
        results = model(str(img_path), conf=CONF_THRESH, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                label = model.names[cls]
                detections.append({"class": label, "conf": round(conf, 3)})

        status = f"{len(detections)} detection(s)" if detections else "no detections"
        print(f"  {img_path.name:<40} → {status}")
        for d in detections:
            print(f"      {d['class']:<20} conf={d['conf']:.3f}")

        results_summary.append({"image": img_path.name, "detections": detections})

        # save annotated image
        out_dir = source_dir / "test_output"
        out_dir.mkdir(exist_ok=True)
        results[0].save(filename=str(out_dir / img_path.name))

    # overall confidence stats
    all_confs = [d["conf"] for r in results_summary for d in r["detections"]]
    if all_confs:
        avg_conf = sum(all_confs) / len(all_confs)
        print(f"\nTotal detections : {len(all_confs)}")
        print(f"Avg confidence   : {avg_conf:.3f} ({avg_conf*100:.1f}%)")
        print(f"Min confidence   : {min(all_confs):.3f}")
        print(f"Max confidence   : {max(all_confs):.3f}")

    report_path = source_dir / "test_output" / "report.json"
    with open(report_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nAnnotated images + report saved to: {out_dir}/")


def mode_video(model, source_video):
    """Run inference on a video and show overlay (press Q to quit)."""
    if not os.path.exists(source_video):
        print(f"[ERROR] Video not found: {source_video}")
        return

    cap = cv2.VideoCapture(source_video)
    total_frames = 0
    total_detections = 0
    class_counts = {name: 0 for name in model.names.values()}

    print(f"\n===== VIDEO INFERENCE: {source_video} =====")
    print("Press Q to stop early.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        results = model(frame, conf=CONF_THRESH, verbose=False)

        for r in results:
            for box in r.boxes:
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                label = model.names[cls]
                total_detections += 1
                class_counts[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.putText(frame, f"Frame {total_frames}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Model Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n===== VIDEO RESULTS =====")
    print(f"Frames processed : {total_frames}")
    print(f"Total detections : {total_detections}")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name:<20} {count} detections")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["info", "val", "images", "video"],
                        default="info")
    parser.add_argument("--data",   help="Path to data.yaml  (for --mode val)")
    parser.add_argument("--source", help="Image folder or video file path")
    args = parser.parse_args()

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    if args.mode == "info":
        mode_info(model)

    elif args.mode == "val":
        if not args.data:
            print("[ERROR] --data <path/to/data.yaml> is required for val mode")
        else:
            mode_val(model, args.data)

    elif args.mode == "images":
        if not args.source:
            print("[ERROR] --source <folder> is required for images mode")
        else:
            mode_images(model, args.source)

    elif args.mode == "video":
        if not args.source:
            print("[ERROR] --source <video.mp4> is required for video mode")
        else:
            mode_video(model, args.source)
