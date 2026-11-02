import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

async function openPdf(pdfFile) {
  const data = new Uint8Array(await pdfFile.arrayBuffer());
  return pdfjs.getDocument({ data });
}

async function renderPage(page, scale) {
  const viewport = page.getViewport({ scale });
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(viewport.width);
  canvas.height = Math.ceil(viewport.height);
  await page.render({ canvas, canvasContext: canvas.getContext("2d"), viewport }).promise;
  return canvas;
}

/** Render every page in a PDF to its own canvas, in document order. */
export async function pdfToCanvases(pdfFile, scale = 1.5, onPageRendered = null) {
  const loadingTask = await openPdf(pdfFile);
  let pdf;
  const canvases = [];

  try {
    pdf = await loadingTask.promise;
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      const page = await pdf.getPage(pageNumber);
      canvases.push(await renderPage(page, scale));
      page.cleanup();
      onPageRendered?.(pageNumber, pdf.numPages);
    }
    return canvases;
  } catch (error) {
    canvases.forEach((canvas) => {
      canvas.width = 0;
      canvas.height = 0;
    });
    throw error;
  } finally {
    if (pdf) pdf.cleanup();
    await loadingTask.destroy();
  }
}

/** Render and consume PDF pages sequentially without retaining the whole document. */
export async function forEachPdfPage(pdfFile, onPage, scale = 1.5) {
  const loadingTask = await openPdf(pdfFile);
  let pdf;

  try {
    pdf = await loadingTask.promise;
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      const page = await pdf.getPage(pageNumber);
      const canvas = await renderPage(page, scale);
      try {
        await onPage(canvas, pageNumber, pdf.numPages);
      } finally {
        canvas.width = 0;
        canvas.height = 0;
        page.cleanup();
      }
    }
  } finally {
    if (pdf) pdf.cleanup();
    await loadingTask.destroy();
  }
}

/** Render one page when retrying or manually correcting a PDF page. */
export async function pdfPageToCanvas(pdfFile, pageNumber, scale = 1.5) {
  const loadingTask = await openPdf(pdfFile);
  let pdf;
  try {
    pdf = await loadingTask.promise;
    if (pageNumber < 1 || pageNumber > pdf.numPages) {
      throw new Error(`PDF page ${pageNumber} no longer exists`);
    }
    const page = await pdf.getPage(pageNumber);
    const canvas = await renderPage(page, scale);
    page.cleanup();
    return canvas;
  } finally {
    if (pdf) pdf.cleanup();
    await loadingTask.destroy();
  }
}