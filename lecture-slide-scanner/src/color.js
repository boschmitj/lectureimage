// Color sampling and color-shift helpers.
// Uses `window.cv` (OpenCV.js).

/**
 * Sample RGBA colors at random points along the contour's inner edge.
 */
export function getRandomEdgeColorsFromContour(
  srcImg,
  singleContour,
  sampleCount = 10,
  maxDistanceInPixels = 5
) {
  const cv = window.cv;
  let mask = null;
  let contoursVec = null;

  try {
    mask = new cv.Mat.zeros(srcImg.rows, srcImg.cols, cv.CV_8UC1);
    contoursVec = new cv.MatVector();
    contoursVec.push_back(singleContour);

    cv.drawContours(mask, contoursVec, 0, new cv.Scalar(255), -1);

    const rect = cv.boundingRect(singleContour);
    const validPoints = [];

    for (let y = rect.y; y < rect.y + rect.height; y++) {
      for (let x = rect.x; x < rect.x + rect.width; x++) {
        if (mask.ucharPtr(y, x)[0] === 255) {
          const dist = cv.pointPolygonTest(
            singleContour,
            new cv.Point(x, y),
            true
          );

          if (dist > 0 && dist <= maxDistanceInPixels) {
            validPoints.push({ x, y, dist });
          }
        }
      }
    }

    const sampledColors = [];
    const numToPick = Math.min(sampleCount, validPoints.length);

    for (let i = 0; i < numToPick; i++) {
      const randomIndex = Math.floor(Math.random() * validPoints.length);
      const pt = validPoints.splice(randomIndex, 1)[0];
      const pixel = srcImg.ucharPtr(pt.y, pt.x);

      sampledColors.push({ point: pt, color: Array.from(pixel) });
    }

    return sampledColors;
  } finally {
    if (mask) mask.delete();
    if (contoursVec) contoursVec.delete();
  }
}

/**
 * Sample RGB colors from random points on the four edges of an image.
 */
export function getRandomEdgeColorsFromCroppedImage(
  srcImg,
  sampleCount = 40,
  edgeWidth = 6,
  offset = 4
) {
  const validPoints = [];
  const sampledColors = [];
  const width = srcImg.cols;
  const height = srcImg.rows;

  for (let i = 0; i < sampleCount; i++) {
    let x, y;
    const edge = Math.floor(Math.random() * 4);

    switch (edge) {
      case 0: // Top
        x = Math.floor(Math.random() * width);
        y = Math.floor(Math.random() * edgeWidth + offset);
        break;
      case 1: // Bottom
        x = Math.floor(Math.random() * width);
        y = height - 1 - Math.floor(Math.random() * edgeWidth + offset);
        break;
      case 2: // Left
        x = Math.floor(Math.random() * edgeWidth + offset);
        y = Math.floor(Math.random() * height);
        break;
      case 3: // Right
        x = width - 1 - Math.floor(Math.random() * edgeWidth + offset);
        y = Math.floor(Math.random() * height);
        break;
    }

    validPoints.push({ x, y });
  }

  for (const pt of validPoints) {
    const pixel = srcImg.ucharPtr(pt.y, pt.x);
    sampledColors.push({ point: pt, color: Array.from(pixel).slice(0, 3) });
  }

  return sampledColors;
}

export function getAverageColor(samples) {
  if (!samples || samples.length === 0) return [0, 0, 0];

  const channelCount = samples[0].color.length;
  const sums = new Array(channelCount).fill(0);

  for (const sample of samples) {
    for (let c = 0; c < channelCount; c++) {
      sums[c] += sample.color[c];
    }
  }

  return sums.map((sum) => Math.round(sum / samples.length));
}

export function getEuclideanColorDistance(colorA, colorB) {
  const rDiff = colorA[0] - colorB[0];
  const gDiff = colorA[1] - colorB[1];
  const bDiff = colorA[2] - colorB[2];
  return Math.sqrt(rDiff * rDiff + gDiff * gDiff + bDiff * bDiff);
}

/**
 * Estimate the base (background) color from edge samples, rejecting outliers.
 */
export function getBaseColor(samples, threshold = 40) {
  if (samples.length === 0) {
    throw new Error("No color samples provided.");
  }

  const channelCount = Math.min(3, ...samples.map((s) => s.color.length));

  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  const medianColor = Array.from({ length: channelCount }, (_, c) =>
    median(samples.map((s) => s.color[c]))
  );

  const filteredSamples = samples.filter((sample) => {
    const distance = Math.sqrt(
      medianColor.reduce(
        (sum, medianValue, c) => sum + (sample.color[c] - medianValue) ** 2,
        0
      )
    );
    return distance <= threshold;
  });

  if (filteredSamples.length === 0) {
    throw new Error("No samples remain after outlier removal.");
  }

  return Array.from({ length: channelCount }, (_, c) => {
    const sum = filteredSamples.reduce(
      (total, sample) => total + sample.color[c],
      0
    );
    return Math.round(sum / filteredSamples.length);
  });
}

/**
 * Shift each channel of srcImg by (targetColor - baseColor), clamping to [0, 255].
 */
export function shiftImageColor(
  srcImg,
  baseColor,
  targetColor = [240, 240, 240]
) {
  const cv = window.cv;
  const dst = new cv.Mat();

  const offsets = targetColor.map((target, i) => target - baseColor[i]);

  const offsetMat = new cv.Mat(
    srcImg.rows,
    srcImg.cols,
    srcImg.type(),
    new cv.Scalar(offsets[0], offsets[1], offsets[2], 0)
  );

  srcImg.convertTo(dst, -1, 1, 0);
  cv.add(srcImg, offsetMat, dst);
  offsetMat.delete();

  return dst;
}
