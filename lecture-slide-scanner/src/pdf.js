// PDF generation using jspdf.
// Produces a landscape PDF with one processed image per page (no rotation).
import { jsPDF } from "jspdf";

/**
 * Generate and download a landscape PDF containing one processed image per page.
 *
 * @param {Object} options
 * @param {string[]} options.images   JPEG data URLs, one per slide
 * @param {Object} options.params     processing parameters (outputWidth/height)
 */
export async function generatePDF({ images, params }) {
  const { outputWidth = 1280, outputHeight = 720 } = params;

  // Landscape page sized exactly to the processed slide dimensions so nothing
  // is rotated, stretched, or cut off.
  const doc = new jsPDF({
    unit: "px",
    orientation: "l",
    format: [outputWidth, outputHeight],
  });

  images.forEach((dataUrl, index) => {
    if (index > 0) {
      doc.addPage([outputWidth, outputHeight], "l");
    }
    // Fill the whole landscape page with the processed image.
    doc.addImage(dataUrl, "JPEG", 0, 0, outputWidth, outputHeight);
  });

  const blob = doc.output("blob");
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "correctified-slides.pdf";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
