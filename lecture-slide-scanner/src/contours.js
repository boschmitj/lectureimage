// Contour detection and quadrilateral ordering.
// Uses `window.cv` (OpenCV.js).
import { getEuclideanColorDistance, getAverageColor, getRandomEdgeColorsFromContour } from "./color.js";

/**
 * Order 4 points clockwise: top-left, top-right, bottom-right, bottom-left.
 * Returns a flat array [x1, y1, x2, y2, x3, y3, x4, y4].
 */
export function orderQuadPoints(pts) {
  const ordered = new Array(4);

  const sums = pts.map((p) => p.x + p.y);
  const diffs = pts.map((p) => p.y - p.x);

  ordered[0] = pts[sums.indexOf(Math.min(...sums))]; // top-left
  ordered[1] = pts[diffs.indexOf(Math.min(...diffs))]; // top-right
  ordered[2] = pts[sums.indexOf(Math.max(...sums))]; // bottom-right
  ordered[3] = pts[diffs.indexOf(Math.max(...diffs))]; // bottom-left

  return ordered.flatMap((p) => [p.x, p.y]);
}

/**
 * Extract { x, y } points from a contour Mat (int32 coordinates).
 */
export function getContourVertices(contourMat) {
  const points = [];
  const data = contourMat.data32S;

  for (let i = 0; i < data.length; i += 2) {
    points.push({ x: data[i], y: data[i + 1] });
  }

  return points;
}

function computeMinArea(minAreaFrac, image) {
  return image.rows * image.cols * minAreaFrac;
}

function computePolygonCandidates(contours, poly, sourceImg, criteria) {
  const cv = window.cv;
  const { minArea, trgtColor, cannyLow, cannyHigh } = criteria;
  const candidates = [];

  for (let i = 0; i < contours.size(); ++i) {
    let tmp = null;
    let cntr = null;

    try {
      tmp = new cv.Mat();
      cntr = contours.get(i);
      const arcLength = cv.arcLength(cntr, true);

      cv.approxPolyDP(cntr, tmp, arcLength * 0.1, true);

      const numVertices = tmp.rows;
      const polyArea = cv.contourArea(tmp);

      if (numVertices === 4 && cv.isContourConvex(tmp) && polyArea > minArea) {
        poly.push_back(tmp);
        const clrDist = getEuclideanColorDistance(
          trgtColor,
          getAverageColor(getRandomEdgeColorsFromContour(sourceImg, tmp, 10, 5))
        );
        candidates.push({
          contour: tmp.clone(),
          area: polyArea,
          colorDistance: clrDist,
          score: null,
        });
      }
    } finally {
      if (cntr) cntr.delete();
      if (tmp) tmp.delete();
    }
  }

  return candidates;
}

function getContourMaxCharacteristics(candidates) {
  let maxArea = null;
  let maxColorDistance = null;

  for (const candidate of candidates) {
    if (!maxArea || candidate.area > maxArea) {
      maxArea = candidate.area;
    }
    if (!maxColorDistance || candidate.colorDistance > maxColorDistance) {
      maxColorDistance = candidate.colorDistance;
    }
  }

  return [maxArea, maxColorDistance];
}

function assignCandidateScores(candidates, maxArea, maxColorDistance) {
  const areaContribution = 0.7;

  for (const candidate of candidates) {
    const areaScore = candidate.area / maxArea;
    const colorScore = 1 - candidate.colorDistance / maxColorDistance;
    const score =
      areaScore * areaContribution + (1 - areaContribution) * colorScore;

    candidate.score = score;
  }
}

/**
 * Select the candidate contour with the highest combined area + color score.
 */
export function selectBestContour(candidates) {
  const [maxArea, maxColorDistance] = getContourMaxCharacteristics(candidates);
  assignCandidateScores(candidates, maxArea, maxColorDistance);

  const bestContour = candidates.reduce((prev, current) => {
    if (!prev) {
      return current;
    }
    return prev.score > current.score ? prev : current;
  });

  return bestContour.contour;
}

/**
 * Find the best convex quadrilateral contour (the slide boundary) in the image.
 *
 * @param {cv.Mat} sourceImg        original image
 * @param {cv.Mat} preprocessedImg  grayscale + blurred image
 * @param {Object} opts             user-controllable detection parameters
 * @param {number} opts.minAreaFrac minimum contour area as fraction of image
 * @param {number} opts.cannyLow    Canny hysteresis low threshold
 * @param {number} opts.cannyHigh   Canny hysteresis high threshold
 */
export function findBestContour(sourceImg, preprocessedImg, opts = {}) {
  const cv = window.cv;
  const trgtColor = [255, 255, 255];

  const minAreaFrac = opts.minAreaFrac ?? 0.00001;
  const cannyLow = opts.cannyLow ?? 50;
  const cannyHigh = opts.cannyHigh ?? 100;

  let cannied = null;
  let contours = null;
  let hierarchy = null;
  let dst = null;
  let poly = null;
  let singleContourVec = null;
  let emptyHierarchy = null;

  try {
    dst = cv.Mat.zeros(
      preprocessedImg.rows,
      preprocessedImg.cols,
      cv.CV_8UC3
    );
    cannied = new cv.Mat();
    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    poly = new cv.MatVector();

    const minArea = computeMinArea(minAreaFrac, preprocessedImg);

    cv.Canny(preprocessedImg, cannied, cannyLow, cannyHigh, 3, false);
    cv.findContours(
      cannied,
      contours,
      hierarchy,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE
    );

    const candidates = computePolygonCandidates(contours, poly, sourceImg, {
      minArea,
      trgtColor,
    });

    const bestContour = selectBestContour(candidates);

    singleContourVec = new cv.MatVector();
    singleContourVec.push_back(bestContour);

    const color = new cv.Scalar(
      Math.round(Math.random() * 255),
      Math.round(Math.random() * 255),
      Math.round(Math.random() * 255)
    );
    emptyHierarchy = new cv.Mat();
    cv.drawContours(
      dst,
      singleContourVec,
      0,
      color,
      1,
      8,
      emptyHierarchy,
      0
    );

    return bestContour;
  } finally {
    if (cannied) cannied.delete();
    if (contours) contours.delete();
    if (hierarchy) hierarchy.delete();
    if (dst) dst.delete();
    if (poly) poly.delete();
    if (singleContourVec) singleContourVec.delete();
    if (emptyHierarchy) emptyHierarchy.delete();
  }
}
