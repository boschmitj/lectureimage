// Image preprocessing helpers (grayscale, blur, perspective, CLAHE, sharpening).
// All functions use `window.cv` (OpenCV.js), which is initialized by index.html.

/**
 * Convert to grayscale and apply a Gaussian blur.
 * Returns a new cv.Mat (caller must delete it).
 */
export function preprocess(image) {
  const cv = window.cv;
  let gray = null;
  let blurred = null;

  try {
    gray = new cv.Mat();
    blurred = new cv.Mat();

    cv.cvtColor(image, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

    return blurred;
  } finally {
    if (gray) gray.delete();
  }
}

/**
 * Apply a perspective warp to obtain a top-down, rectangular view of the slide.
 * `orderedCoords` is a flat array [x1, y1, x2, y2, ...] in TL, TR, BR, BL order.
 */
export function correctPerspective(sourceImage, orderedCoords, width, height) {
  const cv = window.cv;
  let perspectived = null;
  let srcTri = null;
  let dstTri = null;
  let M = null;

  try {
    perspectived = new cv.Mat();
    const dsize = new cv.Size(width, height);
    const outputDims = [0, 0, width, 0, width, height, 0, height];

    srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, orderedCoords);
    dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, outputDims);
    M = cv.getPerspectiveTransform(srcTri, dstTri);

    cv.warpPerspective(
      sourceImage,
      perspectived,
      M,
      dsize,
      cv.INTER_LINEAR,
      cv.BORDER_CONSTANT,
      new cv.Scalar()
    );

    return perspectived;
  } finally {
    if (M) M.delete();
    if (srcTri) srcTri.delete();
    if (dstTri) dstTri.delete();
  }
}

/**
 * Sharpen an image using an unsharp mask.
 */
export function unsharpMask(srcImg, sigma = 1.0, strength = 1.5, kernelSize = [5, 5]) {
  const cv = window.cv;
  const blurred = new cv.Mat();
  const sharpened = new cv.Mat();
  const ksize = new cv.Size(kernelSize[0], kernelSize[1]);

  try {
    cv.GaussianBlur(srcImg, blurred, ksize, sigma, sigma, cv.BORDER_DEFAULT);

    const alpha = 1.0 + strength;
    const beta = -strength;
    const gamma = 0;

    cv.addWeighted(srcImg, alpha, blurred, beta, gamma, sharpened);

    return sharpened;
  } catch (error) {
    sharpened.delete();
    throw error;
  } finally {
    blurred.delete();
  }
}

/**
 * Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L
 * channel of the Lab color space, then convert back to RGBA.
 */
export function applyCLAHE(srcImg, clipLimit = 2, tileSize = 8) {
  const cv = window.cv;
  const rgb = new cv.Mat();
  const lab = new cv.Mat();
  const labChannels = new cv.MatVector();
  const srcChannels = new cv.MatVector();
  const merged = new cv.Mat();
  const result = new cv.Mat();
  const clahe = new cv.CLAHE(clipLimit, new cv.Size(tileSize, tileSize));
  const rgbChannels = new cv.MatVector();
  const rgbaChannels = new cv.MatVector();

  let enhancedL = null;
  let alpha = null;
  let r = null;
  let g = null;
  let b = null;

  try {
    cv.split(srcImg, srcChannels);
    alpha = srcChannels.get(3);

    cv.cvtColor(srcImg, rgb, cv.COLOR_RGBA2RGB);
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);

    cv.split(lab, labChannels);

    const L = labChannels.get(0);
    enhancedL = new cv.Mat();

    try {
      clahe.apply(L, enhancedL);
    } finally {
      L.delete();
    }

    labChannels.set(0, enhancedL);

    cv.merge(labChannels, merged);
    cv.cvtColor(merged, rgb, cv.COLOR_Lab2RGB);

    cv.split(rgb, rgbChannels);
    r = rgbChannels.get(0);
    g = rgbChannels.get(1);
    b = rgbChannels.get(2);

    rgbaChannels.push_back(r);
    rgbaChannels.push_back(g);
    rgbaChannels.push_back(b);
    rgbaChannels.push_back(alpha);

    cv.merge(rgbaChannels, result);

    return result;
  } catch (error) {
    result.delete();
    throw error;
  } finally {
    srcChannels.delete();
    lab.delete();
    labChannels.delete();
    merged.delete();
    clahe.delete();
    rgbChannels.delete();
    rgbaChannels.delete();

    if (enhancedL) {
      enhancedL.delete();
    }
  }
}

/**
 * Simple contrast/scale adjustment: dst = convertScaleAbs(src, alpha, beta).
 */
export function adjustContrast(srcImage, dst, alpha = 1.5, beta = 0) {
  const cv = window.cv;
  cv.convertScaleAbs(srcImage, dst, alpha, beta);
}
