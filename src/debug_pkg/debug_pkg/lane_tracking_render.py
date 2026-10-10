"""Draw published detections and paths without computing driving targets."""

import cv2
import numpy as np


COLORS = {'lane1': (255, 140, 60), 'lane2': (60, 200, 60),
          'solid_line': (245, 245, 245), 'dashed_line': (0, 220, 255),
          'stop_zone': (220, 80, 190), 'traffic_light': (20, 150, 255)}
PATH_COLOR = (0, 0, 255)
TARGET_COLOR = (0, 255, 255)
SOURCE = np.float32([[238, 316], [402, 313], [501, 476], [155, 476]])
DESTINATION = np.float32([[192, 0], [448, 0], [448, 480], [192, 480]])
TRANSFORM = cv2.getPerspectiveTransform(SOURCE, DESTINATION)
INVERSE = cv2.getPerspectiveTransform(DESTINATION, SOURCE)


def points_array(points):
    values = np.asarray(points, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 2
            or not np.all(np.isfinite(values))):
        return np.empty((0, 2), dtype=float)
    return values.copy()


def camera_points(points, width=640, height=480):
    """Map the published 640x180 ROI path back onto the camera image."""
    values = points_array(points)
    if not len(values):
        return values
    values[:, 1] += 300  # Exactly the crop used by the driving extractor.
    mapped = cv2.perspectiveTransform(np.float32([values]), INVERSE)[0]
    return mapped * [width / 640.0, height / 480.0]


def draw_path(image, path, targets):
    path = points_array(path)
    if len(path) >= 2:
        cv2.polylines(image, [np.rint(path).astype(np.int32)], False,
                      PATH_COLOR, 3, cv2.LINE_AA)
    for x, y in points_array(targets):
        cv2.circle(image, (int(round(x)), int(round(y))), 4, TARGET_COLOR, -1)
    return image


def add_legend(image, count, path, status):
    output = cv2.copyMakeBorder(image, 68, 0, 0, 0, cv2.BORDER_CONSTANT,
                               value=(24, 24, 24))
    for name, x in [('lane1', 8), ('lane2', 120), ('solid_line', 252), ('dashed_line', 410)]:
        cv2.putText(output, name, (x, 17), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    COLORS[name], 1, cv2.LINE_AA)
    cv2.putText(output, 'RED: published path   YELLOW: lane target points', (8, 37),
                cv2.FONT_HERSHEY_SIMPLEX, .43, (220, 220, 220), 1, cv2.LINE_AA)
    text = f'YOLO: {count} detections | path: {len(points_array(path))} points | {status}'
    cv2.putText(output, text, (8, 58), cv2.FONT_HERSHEY_SIMPLEX, .4,
                (220, 220, 220), 1, cv2.LINE_AA)
    return output


def render_views(frame, records, path=(), targets=(), status=''):
    """Return camera and ROI overlays; never modify input pixels or points.

    Records contain name, score, height, width, polygon, bbox (x0,y0,x1,y1).
    Every already-published detection is shown, with no new confidence threshold.
    """
    height, width = frame.shape[:2]
    camera = frame.copy()
    roi = cv2.warpPerspective(cv2.resize(frame, (640, 480)), TRANSFORM,
                              (640, 480))[300:].copy()
    # Draw lane areas before narrow markings so boundaries remain legible.
    ordered = sorted(records, key=lambda r: r['name'] not in ('lane1', 'lane2'))
    for record in ordered:
        name, score = record['name'], record['score']
        color = COLORS.get(name, (180, 180, 180))
        polygon = points_array(record['polygon'])
        record_width, record_height = record['width'] or width, record['height'] or height
        if len(polygon) >= 3 and record_width > 0 and record_height > 0:
            canonical = polygon * [640.0 / record_width, 480.0 / record_height]
            raw = polygon * [width / record_width, height / record_height]
            bird = cv2.perspectiveTransform(np.float32([canonical]), TRANSFORM)[0]
            bird[:, 1] -= 300
            for image, shape in ((camera, raw), (roi, bird)):
                contour = np.rint(shape).astype(np.int32)
                layer = image.copy()
                cv2.fillPoly(layer, [contour], color)
                cv2.addWeighted(layer, .3, image, .7, 0, image)
                cv2.polylines(image, [contour], True, color, 2, cv2.LINE_AA)
        bbox = record.get('bbox')
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            scale = np.array([width / max(record_width, 1), height / max(record_height, 1)])
            first = np.rint(np.array([x0, y0]) * scale).astype(int)
            last = np.rint(np.array([x1, y1]) * scale).astype(int)
            cv2.rectangle(camera, tuple(first), tuple(last), color, 1)
        elif len(polygon):
            first = np.rint(np.min(polygon, axis=0) *
                            [width / max(record_width, 1), height / max(record_height, 1)]).astype(int)
        else:
            continue
        label_at = (int(np.clip(first[0], 0, max(0, width-120))),
                    int(np.clip(first[1]-5, 14, height-5)))
        cv2.putText(camera, f'{name} {score:.2f}', label_at,
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    draw_path(roi, path, targets)
    draw_path(camera, camera_points(path, width, height),
              camera_points(targets, width, height))
    return (add_legend(camera, len(records), path, status),
            add_legend(roi, len(records), path, status))
