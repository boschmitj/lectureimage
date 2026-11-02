// Image processing pipeline with user-controllable parameters.
// All functions use `window.cv` (OpenCV.js), which is initialized by index.html.
import {
  preprocess,
  correctPerspective,
  unsharpMask,
  applyCLAHE,
} from "./preprocessing.js";
import {
  findBestContour,
  getContourVertices,
  orderQuadPoints,
} from "./contours.js";
import {
  getRandomEdgeColorsFromCroppedImage,
  getBaseColor,
  shiftImageColor,
} from "./color.js";

// Default processing parameters. These are exposed in the settings sidebar so
// the user can control how each slide is processed.
//
// NOTE: minAreaFrac is expressed as a fraction of the image area. The UI shows
// it as a percentage (0.01 = 1%).
export const DEFAULT_PARAMS = {
  outputWidth: 1280,
  outputHeight: 720,
  cannyLow: 50,
  cannyHigh: 100,
  minAreaFrac: 0.01,
  claheClipLimit: 1.5,
  claheTileSize: 8,
  sharpenSigma: 0.8,
  sharpenStrength: 0.5,
  sharpenKernel: [5, 5],
  colorCorrection: false,
  clahe: false,
  sharpen: true,
};

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${file.name}`));
    img.src = URL.createObjectURL(file);
  });
}

/**
 * Run the full document/slide scanning pipeline on `inputCanvas` and render the
 * result to `outputCanvas`. Returns a data URL of the processed image.
 *
 * @param {HTMLCanvasElement} inputCanvas   canvas holding the source image
 * @param {HTMLCanvasElement} outputCanvas  canvas the result is drawn into
 * @param {Object} params                   user-controlled processing parameters
 * @param {number[]} manualCorners          optional [x1,y1,x2,y2,x3,y3,x4,y4]
 *                                          in TL,TR,BR,BL order (original px).
 *                                          When set, skips auto contour detection.
 * @returns {Promise<string>}               JPEG data URL of the processed slide
 */
export async function processImage(inputCanvas, outputCanvas, params = {}, manualCorners = null) {
  const p = { ...DEFAULT_PARAMS, ...params };
  const cv = window.cv;
  const sourceImg = cv.imread(inputCanvas);

  const mats = [sourceImg];

  try {
    // 1. Grayscale + Gaussian blur
    const preprocessed = preprocess(sourceImg);
    mats.push(preprocessed);

    let orderedCoords;
    if (manualCorners && manualCorners.length === 8) {
      // Use user-provided corners (TL, TR, BR, BL), skipping auto detection.
      orderedCoords = manualCorners;
    } else {
      // 2. Detect the slide boundary (best convex quadrilateral contour)
      const matchedContour = findBestContour(sourceImg, preprocessed, {
        minAreaFrac: p.minAreaFrac,
        cannyLow: p.cannyLow,
        cannyHigh: p.cannyHigh,
      });
      mats.push(matchedContour);

      const matchedContourCoords = getContourVertices(matchedContour);
      orderedCoords = orderQuadPoints(matchedContourCoords);
    }

    // 3. Perspective correction to the user-controlled output size
    const perspectived = correctPerspective(
      sourceImg,
      orderedCoords,
      p.outputWidth,
      p.outputHeight
    );
    mats.push(perspectived);

    // 4. Optional: estimate base color from edges and neutralize it
    let postprocessed = perspectived;
    if (p.colorCorrection) {
      const samples = getRandomEdgeColorsFromCroppedImage(perspectived, 40);
      const baseColor = getBaseColor(samples);
      postprocessed = shiftImageColor(perspectived, baseColor);
      mats.push(postprocessed);
    }

    // 5. Optional: contrast enhancement (CLAHE)
    let working = postprocessed;
    if (p.clahe) {
      working = applyCLAHE(postprocessed, p.claheClipLimit, p.claheTileSize);
      mats.push(working);
    }

    // 6. Optional: unsharp-mask sharpening
    let sharpened = working;
    if (p.sharpen) {
      sharpened = unsharpMask(
        working,
        p.sharpenSigma,
        p.sharpenStrength,
        p.sharpenKernel
      );
      mats.push(sharpened);
    }

    // 7. Render result to the output canvas and return a data URL
    cv.imshow(outputCanvas, sharpened);
    return outputCanvas.toDataURL("image/jpeg", 0.92);
  } finally {
    mats.forEach((m) => {
      if (m) m.delete();
    });
  }
}

/**
 * Run the scanner on a canvas and return the processed image plus its original.
 * @param {HTMLCanvasElement} inputCanvas
 * @param {Object} params
 * @param {number[]} manualCorners
 */
export async function processImageCanvas(inputCanvas, params = {}, manualCorners = null) {
  const originalDataUrl = inputCanvas.toDataURL("image/jpeg", 0.92);
  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = 1;
  outputCanvas.height = 1;

  const dataUrl = await processImage(inputCanvas, outputCanvas, params, manualCorners);

  return {
    dataUrl,
    originalDataUrl,
    width: inputCanvas.width,
    height: inputCanvas.height,
  };
}

/**
 * Load a single image file, run the pipeline, and return the processed result
 * plus the original image data URL for before/after comparison.
 *
 * @param {File} file    image file to process
 * @param {Object} params user-controlled processing parameters
 * @returns {Promise<{dataUrl: string, originalDataUrl: string, fileName: string, width: number, height: number}>}
 */
export async function processImageFile(file, params = {}, manualCorners = null, onOriginal = null) {
  const img = await loadImageFromFile(file);

  const inputCanvas = document.createElement("canvas");
  inputCanvas.width = img.naturalWidth;
  inputCanvas.height = img.naturalHeight;
  inputCanvas.getContext("2d").drawImage(img, 0, 0);
  if (onOriginal) onOriginal(inputCanvas.toDataURL("image/jpeg", 0.92));

  URL.revokeObjectURL(img.src);
  return {
    ...await processImageCanvas(inputCanvas, params, manualCorners),
    fileName: file.name,
  };
}
