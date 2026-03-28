from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class DetectionPoint:
    frame_index: int
    timestamp_sec: float
    motion_ratio: float
    largest_contour_area: float


@dataclass
class ClipCandidate:
    clip_id: str
    source_video: Path
    output_path: Path
    start_sec: float
    end_sec: float
    active_start_sec: float
    active_end_sec: float
    score: float
    peak_motion_ratio: float
    peak_contour_area: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect likely punch exchanges in a boxing video and cut them into "
            "short review clips using OpenCV."
        )
    )
    parser.add_argument("input_video", type=Path, help="Path to the source fight video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "candidate_clips",
        help="Directory where clips and the manifest should be written",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional CSV path. Defaults to <output-dir>/<video-stem>_manifest.csv",
    )
    parser.add_argument(
        "--analysis-fps",
        type=float,
        default=10.0,
        help="Frame rate used for motion analysis. Lower values are faster.",
    )
    parser.add_argument(
        "--analysis-max-width",
        type=int,
        default=320,
        help=(
            "Downscale the ROI to this width before motion analysis. Lower values "
            "are faster but less precise."
        ),
    )
    parser.add_argument(
        "--pre-roll",
        type=float,
        default=0.35,
        help="Seconds to include before detected motion",
    )
    parser.add_argument(
        "--post-roll",
        type=float,
        default=0.55,
        help="Seconds to include after detected motion",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.45,
        help="Merge detections that are within this many seconds of each other",
    )
    parser.add_argument(
        "--min-clip-length",
        type=float,
        default=0.9,
        help="Minimum output clip length in seconds",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Optional cap on the number of clips to export",
    )
    parser.add_argument(
        "--clip-max-width",
        type=int,
        default=960,
        help=(
            "Downscale exported clips to this width for faster writing and easier "
            "manual review. Set to 0 to keep original resolution."
        ),
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="0.10,0.12,0.90,0.92",
        help=(
            "Normalized ROI as x1,y1,x2,y2. Use this to focus on the ring and "
            "ignore borders or score graphics."
        ),
    )
    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=20,
        help="Pixel difference threshold for motion masks",
    )
    parser.add_argument(
        "--motion-ratio-threshold",
        type=float,
        default=0.012,
        help="Minimum fraction of ROI pixels that must be moving",
    )
    parser.add_argument(
        "--max-motion-ratio",
        type=float,
        default=0.18,
        help=(
            "Ignore detections where too much of the ROI changes at once. This helps "
            "filter broadcast cuts, zooms, and big camera moves."
        ),
    )
    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=550.0,
        help="Minimum moving contour area to count as a candidate",
    )
    parser.add_argument(
        "--max-active-duration",
        type=float,
        default=8.0,
        help=(
            "Skip merged candidates whose continuous detected motion lasts longer than "
            "this many seconds."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-candidate details while processing",
    )
    return parser


def parse_roi(raw_roi: str) -> tuple[float, float, float, float]:
    pieces = [piece.strip() for piece in raw_roi.split(",")]
    if len(pieces) != 4:
        raise ValueError("ROI must contain four comma-separated values")

    x1, y1, x2, y2 = (float(piece) for piece in pieces)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("ROI values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")

    return x1, y1, x2, y2


def seconds_to_frame(seconds: float, fps: float) -> int:
    return max(0, int(round(seconds * fps)))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def make_safe_stem(raw_name: str, max_length: int = 48) -> str:
    normalized = unicodedata.normalize("NFKD", raw_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")
    if not ascii_name:
        ascii_name = "video"

    shortened = ascii_name[:max_length].rstrip("_") or "video"
    digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:8]
    return f"{shortened}_{digest}"


def compute_roi_bounds(
    width: int, height: int, roi: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    left = int(round(width * x1))
    top = int(round(height * y1))
    right = int(round(width * x2))
    bottom = int(round(height * y2))

    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))

    return left, top, right, bottom


def resize_to_max_width(frame, max_width: int):
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame, 1.0

    scale = max_width / float(frame.shape[1])
    new_height = max(1, int(round(frame.shape[0] * scale)))
    resized = cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_motion_points(
    input_video: Path,
    analysis_fps: float,
    analysis_max_width: int,
    roi: tuple[float, float, float, float],
    diff_threshold: int,
    motion_ratio_threshold: float,
    max_motion_ratio: float,
    min_contour_area: float,
) -> tuple[list[DetectionPoint], float, int, int, int]:
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(round(fps / max(analysis_fps, 0.1))))

    roi_bounds = compute_roi_bounds(width, height, roi)
    left, top, right, bottom = roi_bounds

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    previous_gray = None
    detections: list[DetectionPoint] = []
    frame_index = -1
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    last_progress_second = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_index += 1
        if frame_index % frame_interval != 0:
            continue

        roi_frame = frame[top:bottom, left:right]
        analysis_frame, analysis_scale = resize_to_max_width(roi_frame, analysis_max_width)
        gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if previous_gray is None:
            previous_gray = gray
            continue

        frame_diff = cv2.absdiff(previous_gray, gray)
        _, motion_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour_area = max(
            (cv2.contourArea(contour) for contour in contours),
            default=0.0,
        )
        motion_pixels = cv2.countNonZero(motion_mask)
        motion_ratio = motion_pixels / float(motion_mask.shape[0] * motion_mask.shape[1])
        adjusted_min_contour_area = min_contour_area * (analysis_scale ** 2)

        if (
            motion_ratio >= motion_ratio_threshold
            and motion_ratio <= max_motion_ratio
            and largest_contour_area >= adjusted_min_contour_area
        ):
            detections.append(
                DetectionPoint(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    motion_ratio=motion_ratio,
                    largest_contour_area=largest_contour_area,
                )
            )

        previous_gray = gray

        current_second = int(frame_index / fps)
        if current_second >= last_progress_second + 10:
            total_seconds = total_frames / fps if fps > 0 else 0.0
            print(
                f"Analyzing video... {current_second:.0f}s / {total_seconds:.0f}s",
                file=sys.stderr,
            )
            last_progress_second = current_second

    capture.release()
    return detections, fps, width, height, total_frames


def merge_detections_into_candidates(
    detections: list[DetectionPoint],
    input_video: Path,
    output_stem: str,
    video_duration_sec: float,
    pre_roll: float,
    post_roll: float,
    merge_gap: float,
    min_clip_length: float,
    max_active_duration: float,
    output_dir: Path,
) -> list[ClipCandidate]:
    if not detections:
        return []

    merged_groups: list[list[DetectionPoint]] = [[detections[0]]]
    for detection in detections[1:]:
        current_group = merged_groups[-1]
        previous_detection = current_group[-1]
        if detection.timestamp_sec - previous_detection.timestamp_sec <= merge_gap:
            current_group.append(detection)
        else:
            merged_groups.append([detection])

    candidates: list[ClipCandidate] = []
    for index, group in enumerate(merged_groups, start=1):
        active_start = group[0].timestamp_sec
        active_end = group[-1].timestamp_sec
        active_duration = active_end - active_start
        if max_active_duration > 0 and active_duration > max_active_duration:
            continue

        start_sec = clamp(active_start - pre_roll, 0.0, video_duration_sec)
        end_sec = clamp(active_end + post_roll, 0.0, video_duration_sec)

        if end_sec - start_sec < min_clip_length:
            needed = min_clip_length - (end_sec - start_sec)
            start_sec = clamp(start_sec - needed / 2.0, 0.0, video_duration_sec)
            end_sec = clamp(end_sec + needed / 2.0, 0.0, video_duration_sec)

            if end_sec - start_sec < min_clip_length:
                start_sec = clamp(end_sec - min_clip_length, 0.0, video_duration_sec)
                end_sec = clamp(start_sec + min_clip_length, 0.0, video_duration_sec)

        peak_motion_ratio = max(point.motion_ratio for point in group)
        peak_contour_area = max(point.largest_contour_area for point in group)
        score = peak_motion_ratio * peak_contour_area
        clip_id = f"{output_stem}_candidate_{index:04d}"
        output_path = output_dir / f"{clip_id}.mp4"
        candidates.append(
            ClipCandidate(
                clip_id=clip_id,
                source_video=input_video,
                output_path=output_path,
                start_sec=start_sec,
                end_sec=end_sec,
                active_start_sec=active_start,
                active_end_sec=active_end,
                score=score,
                peak_motion_ratio=peak_motion_ratio,
                peak_contour_area=peak_contour_area,
            )
        )

    # Expanded windows can overlap after pre-roll/post-roll, so merge once more.
    stitched: list[ClipCandidate] = [candidates[0]]
    for candidate in candidates[1:]:
        previous = stitched[-1]
        if candidate.start_sec <= previous.end_sec:
            previous.end_sec = max(previous.end_sec, candidate.end_sec)
            previous.active_end_sec = max(previous.active_end_sec, candidate.active_end_sec)
            previous.score = max(previous.score, candidate.score)
            previous.peak_motion_ratio = max(
                previous.peak_motion_ratio, candidate.peak_motion_ratio
            )
            previous.peak_contour_area = max(
                previous.peak_contour_area, candidate.peak_contour_area
            )
            continue

        stitched.append(candidate)

    for index, candidate in enumerate(stitched, start=1):
        candidate.clip_id = f"{output_stem}_candidate_{index:04d}"
        candidate.output_path = output_dir / f"{candidate.clip_id}.mp4"

    return stitched


def write_clips(
    input_video: Path,
    candidates: list[ClipCandidate],
    fps: float,
    width: int,
    height: int,
    clip_max_width: int,
) -> None:
    if not candidates:
        return

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video for clip extraction: {input_video}")

    if clip_max_width > 0 and width > clip_max_width:
        output_width = clip_max_width
        output_height = max(1, int(round(height * (clip_max_width / float(width)))))
    else:
        output_width = width
        output_height = height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for candidate in candidates:
        start_frame = seconds_to_frame(candidate.start_sec, fps)
        end_frame = max(start_frame + 1, seconds_to_frame(candidate.end_sec, fps))

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(
            str(candidate.output_path), fourcc, fps, (output_width, output_height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output clip: {candidate.output_path}")

        print(
            f"Writing clip {candidate.clip_id} "
            f"({candidate.start_sec:.2f}s -> {candidate.end_sec:.2f}s)",
            file=sys.stderr,
        )
        current_frame = start_frame
        while current_frame < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if output_width != width or output_height != height:
                frame = cv2.resize(
                    frame, (output_width, output_height), interpolation=cv2.INTER_AREA
                )
            writer.write(frame)
            current_frame += 1

        writer.release()

    capture.release()


def write_manifest(manifest_path: Path, candidates: list[ClipCandidate]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "clip_id",
                "clip_path",
                "source_video",
                "label",
                "start_sec",
                "end_sec",
                "active_start_sec",
                "active_end_sec",
                "duration_sec",
                "score",
                "peak_motion_ratio",
                "peak_contour_area",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "clip_id": candidate.clip_id,
                    "clip_path": str(candidate.output_path),
                    "source_video": str(candidate.source_video),
                    "label": "",
                    "start_sec": f"{candidate.start_sec:.3f}",
                    "end_sec": f"{candidate.end_sec:.3f}",
                    "active_start_sec": f"{candidate.active_start_sec:.3f}",
                    "active_end_sec": f"{candidate.active_end_sec:.3f}",
                    "duration_sec": f"{candidate.end_sec - candidate.start_sec:.3f}",
                    "score": f"{candidate.score:.4f}",
                    "peak_motion_ratio": f"{candidate.peak_motion_ratio:.6f}",
                    "peak_contour_area": f"{candidate.peak_contour_area:.2f}",
                }
            )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_video = args.input_video.resolve()
    if not input_video.exists():
        parser.error(f"Input video does not exist: {input_video}")

    output_stem = make_safe_stem(input_video.stem)
    output_dir = args.output_dir.resolve() / output_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path is not None
        else output_dir / f"{output_stem}_manifest.csv"
    )

    roi = parse_roi(args.roi)
    detections, fps, width, height, total_frames = detect_motion_points(
        input_video=input_video,
        analysis_fps=args.analysis_fps,
        analysis_max_width=args.analysis_max_width,
        roi=roi,
        diff_threshold=args.diff_threshold,
        motion_ratio_threshold=args.motion_ratio_threshold,
        max_motion_ratio=args.max_motion_ratio,
        min_contour_area=args.min_contour_area,
    )

    video_duration_sec = total_frames / fps if fps > 0 else 0.0
    candidates = merge_detections_into_candidates(
        detections=detections,
        input_video=input_video,
        output_stem=output_stem,
        video_duration_sec=video_duration_sec,
        pre_roll=args.pre_roll,
        post_roll=args.post_roll,
        merge_gap=args.merge_gap,
        min_clip_length=args.min_clip_length,
        max_active_duration=args.max_active_duration,
        output_dir=output_dir,
    )

    if args.max_clips is not None:
        candidate_limit = max(args.max_clips, 0)
        ranked_candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
        candidates = sorted(
            ranked_candidates[:candidate_limit],
            key=lambda candidate: candidate.start_sec,
        )

    write_clips(
        input_video=input_video,
        candidates=candidates,
        fps=fps,
        width=width,
        height=height,
        clip_max_width=args.clip_max_width,
    )
    write_manifest(manifest_path, candidates)

    print(f"Source video: {input_video}")
    print(f"Detections found: {len(detections)}")
    print(f"Candidate clips written: {len(candidates)}")
    print(f"Manifest: {manifest_path}")

    if args.verbose:
        for candidate in candidates:
            duration = candidate.end_sec - candidate.start_sec
            print(
                f"{candidate.clip_id}: {candidate.start_sec:.2f}s -> "
                f"{candidate.end_sec:.2f}s ({duration:.2f}s), "
                f"score={candidate.score:.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
