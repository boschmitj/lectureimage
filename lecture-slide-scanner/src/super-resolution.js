// Super-resolution via FSRCNN (ONNX Runtime Web).
// NOTE: FSRCNN model loading is currently disabled.

/**
 * Load the FSRCNN inference session.
 * Disabled for now: returns null instead of loading the model.
 */
export async function loadFSRCNN() {
  console.log("FSRCNN model loading is disabled.");
  return null;
}

/**
 * Upscale a grayscale image 4x with FSRCNN.
 * With FSRCNN disabled, this returns a clone of the input so the pipeline
 * can continue without the model.
 */
export async function superResolveGray(gray) {
  const cv = window.cv;

  if (gray.type() !== cv.CV_8UC1) {
    throw new Error("Expected an 8-bit grayscale cv.Mat.");
  }

  console.warn(
    "FSRCNN super-resolution is disabled; returning the grayscale image unchanged."
  );
  return gray.clone();
}

/**
 * Run a built-in FSRCNN smoke test.
 * Disabled for now because model loading is disabled.
 */
export async function testFSRCNN() {
  console.log("FSRCNN test is disabled because model loading is disabled.");
}
