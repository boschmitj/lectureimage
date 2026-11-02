import { processImageCanvas, processImageFile, DEFAULT_PARAMS } from "./pipeline.js";
import { generatePDF } from "./pdf.js";
import { forEachPdfPage } from "./pdf-input.js";

const $ = (id) => document.getElementById(id);

// DOM
const fileInput = $("fileInput");
const dropZone = $("dropZone");
const fileList = $("fileList");
const progressPanel = $("progressPanel");
const progressLabel = $("progressLabel");
const progressValue = $("progressValue");
const progressBar = $("progressBar");
const btnPdf = $("btnPdf");
const btnRedo = $("btnRedo");
const btnSettings = $("btnSettings");
const btnClose = $("btnCloseSidebar");
const sidebar = $("sidebar");
const backdrop = $("backdrop");
const btnReset = $("btnReset");

// Modal DOM
const modal = $("compareModal");
const modalBackdrop = $("modalBackdrop");
const modalClose = $("modalClose");
const modalFilename = $("modalFilename");
const modalOriginal = $("modalOriginal");
const modalProcessed = $("modalProcessed");
const modalEditCorners = $("modalEditCorners");

// Application state
let settings = { ...DEFAULT_PARAMS };
let jobs = []; // { id, file, status, dataUrl, originalDataUrl, error, manualCorners }
let processed = []; // { dataUrl, originalDataUrl, fileName }
let processing = false;
const PDF_RENDER_SCALE = 1.5;

function updateProgress(label, value = null, max = 1) {
  progressPanel.hidden = false;
  progressLabel.textContent = label;
  if (value === null) {
    progressBar.removeAttribute("value");
    progressValue.textContent = "";
    return;
  }

  progressBar.max = Math.max(max, 1);
  progressBar.value = value;
  progressValue.textContent = `${Math.round(value / Math.max(max, 1) * 100)}%`;
}

// ---------------------------------------------------------------------------
// Settings sidebar
// ---------------------------------------------------------------------------

function openSidebar() {
  sidebar.classList.add("open");
  backdrop.classList.add("visible");
}

function closeSidebar() {
  sidebar.classList.remove("open");
  backdrop.classList.remove("visible");
}

btnSettings.addEventListener("click", openSidebar);
btnClose.addEventListener("click", closeSidebar);
backdrop.addEventListener("click", closeSidebar);

// Sync a slider <-> number input pair for a single setting key
function bindSetting(key) {
  const rangeEl = $(`p-${key}`);
  const numEl = $(`n-${key}`);
  const valEl = $(`v-${key}`);

  if (rangeEl?.type === "checkbox") {
    const setValue = (checked) => {
      settings[key] = checked;
      rangeEl.checked = checked;
    };

    setValue(settings[key] === true);
    rangeEl.onchange = (event) => setValue(event.target.checked);
    return;
  }

  if (!rangeEl || !numEl) return;

  // minAreaFrac is shown as a percentage in the UI (1 = 1%) but stored
  // internally as a fraction (0.01 = 1%) for the pipeline.
  const isPercent = key === "minAreaFrac";
  const toInternal = (v) => (isPercent ? v / 100 : v);
  const toDisplay = (v) => (isPercent ? v * 100 : v);

  const setValue = (v) => {
    v = parseFloat(v);
    if (!isFinite(v)) return;
    settings[key] = toInternal(v);
    rangeEl.value = toDisplay(settings[key]);
    numEl.value = toDisplay(settings[key]);
    if (valEl) valEl.textContent = toDisplay(settings[key]).toString();
  };

  // Initial sync
  setValue(toDisplay(settings[key]));

  rangeEl.oninput = (e) => setValue(e.target.value);
  numEl.oninput = (e) => setValue(e.target.value);
  numEl.onchange = (e) => setValue(e.target.value);
}

function wireControls() {
  Object.keys(settings).forEach(bindSetting);
}

btnReset.addEventListener("click", () => {
  settings = { ...DEFAULT_PARAMS };
  wireControls();
});

// ---------------------------------------------------------------------------
// Modal (before/after)
// ---------------------------------------------------------------------------

function openModal(job) {
  modalFilename.textContent = job.label || job.file.name;
  modalOriginal.src = job.originalDataUrl || "";
  modalProcessed.src = job.dataUrl || "";
  modal.classList.add("open");
}

function closeModal() {
  modal.classList.remove("open");
  modalOriginal.src = "";
  modalProcessed.src = "";
}

modalClose.addEventListener("click", closeModal);
modalBackdrop.addEventListener("click", closeModal);
modal.addEventListener("click", (e) => {
  if (e.target === modal) closeModal();
});

// ---------------------------------------------------------------------------
// selectImageCorners — canvas-based manual corner selection
// ---------------------------------------------------------------------------

function selectImageCorners(image) {
  return new Promise((resolve) => {
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;

    const scale = Math.min(1, 1200 / width, 800 / height);
    const overlay = document.createElement("div");

    overlay.style.cssText = `
        position: fixed;
        inset: 0;
        z-index: 99999;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 16px;
    `;

    overlay.innerHTML = `
        <div style="
            background: white;
            padding: 16px;
            border-radius: 8px;
            max-width: 100%;
            max-height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            font-family: ${getComputedStyle(document.documentElement).fontFamily || 'Geist, sans-serif'};
        ">
            <p style="margin:0; color:#201335; font-weight:600">Drag the four points to the slide corners.</p>
            <canvas style="
                display: block;
                max-width: 100%;
                max-height: 75vh;
                touch-action: none;
                cursor: crosshair;
            "></canvas>
            <div style="display:flex; gap:10px">
                <button data-confirm class="corner-btn-confirm">Use points</button>
                <button data-cancel class="corner-btn-cancel">Cancel</button>
            </div>
        </div>
    `;

    document.body.appendChild(overlay);

    const canvas = overlay.querySelector("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);

    // Order: top-left, top-right, bottom-right, bottom-left.
    const points = [
      { x: width * 0.1, y: height * 0.1 },
      { x: width * 0.9, y: height * 0.1 },
      { x: width * 0.9, y: height * 0.9 },
      { x: width * 0.1, y: height * 0.9 }
    ];

    let dragging = -1;

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

      // Draw the selection polygon.
      ctx.beginPath();
      points.forEach((p, i) => {
        const x = p.x * canvas.width / width;
        const y = p.y * canvas.height / height;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.closePath();

      ctx.fillStyle = "rgba(0, 140, 255, 0.15)";
      ctx.fill();
      ctx.strokeStyle = "#008cff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw draggable corner points.
      points.forEach((p, i) => {
        const x = p.x * canvas.width / width;
        const y = p.y * canvas.height / height;

        ctx.beginPath();
        ctx.arc(x, y, 9, 0, 2 * Math.PI);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.strokeStyle = "#008cff";
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.fillStyle = "#005bbb";
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(i + 1, x, y);
      });
    }

    function getPosition(event) {
      const rect = canvas.getBoundingClientRect();

      return {
        x: Math.max(0, Math.min(width,
          (event.clientX - rect.left) / rect.width * width)),
        y: Math.max(0, Math.min(height,
          (event.clientY - rect.top) / rect.height * height))
      };
    }

    canvas.addEventListener("pointerdown", (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      let nearestDistance = Infinity;

      points.forEach((p, i) => {
        const px = p.x / width * rect.width;
        const py = p.y / height * rect.height;
        const distance = Math.hypot(x - px, y - py);

        if (distance < nearestDistance) {
          nearestDistance = distance;
          dragging = i;
        }
      });

      if (nearestDistance > 25) {
        dragging = -1;
        return;
      }

      canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    canvas.addEventListener("pointermove", (event) => {
      if (dragging === -1) return;

      points[dragging] = getPosition(event);
      draw();
    });

    function stopDragging() {
      dragging = -1;
    }

    canvas.addEventListener("pointerup", stopDragging);
    canvas.addEventListener("pointercancel", stopDragging);

    overlay.querySelector("[data-confirm]").onclick = () => {
      const result = points.map(p => ({
        x: Math.round(p.x),
        y: Math.round(p.y)
      }));

      overlay.remove();
      resolve(result);
    };

    overlay.querySelector("[data-cancel]").onclick = () => {
      overlay.remove();
      resolve(null);
    };

    draw();
  });
}

// ---------------------------------------------------------------------------
// Re-process a single image with manual corners
// ---------------------------------------------------------------------------

async function reprocessJobWithCorners(job) {
  // Create an Image element from the original data URL for corner selection
  const img = new Image();
  img.src = job.originalDataUrl;
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error("Failed to load the original page image"));
  });

  const corners = await selectImageCorners(img);
  if (!corners) return false; // cancelled

  const flatCorners = [
    corners[0].x, corners[0].y,
    corners[1].x, corners[1].y,
    corners[2].x, corners[2].y,
    corners[3].x, corners[3].y
  ];

  try {
    const result = await processJob(job, flatCorners);
    job.dataUrl = result.dataUrl;
    job.originalDataUrl = result.originalDataUrl;
    job.manualCorners = flatCorners;
    job.status = "done";
    job.error = null;

    // Update processed list
    processed = jobs.filter((item) => item.status === "done").map((item) => item.result);

    return true;
  } catch (err) {
    console.error("Corner re-processing failed", err);
    alert("Failed to re-process: " + (err.message || String(err)));
    return false;
  }
}

// ---------------------------------------------------------------------------
// File list rendering
// ---------------------------------------------------------------------------

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function statusLabel(status) {
  const map = {
    pending: "pending",
    processing: "processing\u2026",
    done: "done",
    error: "error",
  };
  return map[status] || status;
}

function renderList() {
  fileList.innerHTML = "";

  jobs.forEach((job) => {
    const card = document.createElement("div");
    card.className = `file-card ${job.status}`;
    card.dataset.id = job.id;
    card.style.cursor = "pointer";

    // Click on card: error opens corner picker directly, done opens modal
    card.addEventListener("click", async () => {
      if (job.status === "error" && !job.loadError) {
        const reProcessed = await reprocessJobWithCorners(job);
        if (reProcessed) {
          renderList();
          openModal(job);
        }
      } else if (job.status !== "error") {
        openModal(job);
      }
    });

    const thumb = document.createElement("div");
    thumb.className = "file-thumb";

    const status = document.createElement("div");
    status.className = "file-status";
    status.innerHTML = `<span>${escapeHtml(statusLabel(job.status))}</span>${
      job.error ? `<span style="color:#fca5a5">!</span>` : ""
    }`;

    const info = document.createElement("div");
    info.className = "file-info";

    const name = document.createElement("div");
    name.className = "file-name";
    name.textContent = escapeHtml(job.label || job.file.name);

    const meta = document.createElement("div");
    meta.className = "file-meta";
    meta.textContent = job.dataUrl ? "processed" : job.file.type || "image";

    thumb.appendChild(status);
    info.appendChild(name);
    info.appendChild(meta);
    card.appendChild(thumb);
    card.appendChild(info);
    fileList.appendChild(card);

    if (job.status === "done" && job.dataUrl) {
      thumb.innerHTML = "";
      const img = document.createElement("img");
      img.src = job.dataUrl;
      img.alt = "Processed slide";
      thumb.appendChild(img);
      thumb.appendChild(status);
    } else if (job.status === "processing") {
      thumb.innerHTML = `<div class="empty">processing\u2026</div>`;
      thumb.appendChild(status);
    } else if (job.status === "error") {
      thumb.innerHTML = `<div class="empty" style="color:#f87171">error</div>`;
      thumb.appendChild(status);
    } else {
      thumb.innerHTML = `<div class="empty">queued</div>`;
      thumb.appendChild(status);
    }
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isImage(file) {
  return file.type && file.type.startsWith("image/");
}

function isPdf(file) {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

function isSupportedFile(file) {
  return isImage(file) || isPdf(file);
}

async function processJob(job, manualCorners = null) {
  if (!isPdf(job.file)) {
    return processImageFile(
      job.file,
      settings,
      manualCorners,
      (originalDataUrl) => { job.originalDataUrl = originalDataUrl; }
    );
  }

  const canvas = job.sourceCanvas || await canvasFromDataUrl(job.originalDataUrl);
  try {
    if (!job.originalDataUrl) job.originalDataUrl = canvas.toDataURL("image/jpeg", 0.92);
    return {
      ...await processImageCanvas(canvas, settings, manualCorners),
      originalDataUrl: job.originalDataUrl,
      fileName: job.label,
    };
  } finally {
    canvas.width = 0;
    canvas.height = 0;
    job.sourceCanvas = null;
  }
}

function canvasFromDataUrl(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      canvas.getContext("2d").drawImage(image, 0, 0);
      resolve(canvas);
    };
    image.onerror = () => reject(new Error("Failed to load the cached PDF page image"));
    image.src = dataUrl;
  });
}

// Recursively collect image files from a directory entry (for folder drops).
async function collectDirFiles(dirEntry, out) {
  const reader = dirEntry.createReader();
  const readAll = () =>
    new Promise((resolve) => reader.readEntries(resolve));
  let entries = [];
  let batch;
  while ((batch = await readAll()).length) {
    entries = entries.concat(batch);
  }

  for (const entry of entries) {
    if (entry.isFile) {
      const file = await new Promise((resolve) => entry.file(resolve));
      out.push(file);
    } else if (entry.isDirectory) {
      await collectDirFiles(entry, out);
    }
  }
}

// Collect image files from DataTransfer items (supports nested folders).
async function collectFilesFromItems(items) {
  const files = [];
  for (const item of items) {
    const entry = item.webkitGetAsEntry();
    if (!entry) continue;
    if (entry.isFile) {
      const file = await new Promise((resolve) => entry.file(resolve));
      files.push(file);
    } else if (entry.isDirectory) {
      await collectDirFiles(entry, files);
    }
  }
  return files;
}

// ---------------------------------------------------------------------------
// Upload handling
// ---------------------------------------------------------------------------

async function handleFiles(fileCollection) {
  const selectedFiles = Array.from(fileCollection);
  const files = selectedFiles.filter(isSupportedFile);
  const invalidFiles = selectedFiles.filter((file) => !isSupportedFile(file));
  if (invalidFiles.length) {
    alert(`Unsupported file type: ${invalidFiles.map((file) => file.name).join(", ")}`);
  }
  if (!files.length) return;

  // Stable order: respect folder-relative paths when available
  files.sort(
    (a, b) => (a.webkitRelativePath || "").localeCompare(b.webkitRelativePath || "")
  );

  processing = true;
  btnPdf.disabled = true;
  btnRedo.disabled = true;
  jobs = [];
  processed = [];
  renderList();
  updateProgress(`Importing ${files.length} file${files.length === 1 ? "" : "s"}`, null);

  async function processAndDisplay(job, progressLabelText) {
    jobs.push(job);
    renderList();
    job.status = "processing";
    updateProgress(progressLabelText, null);
    renderList();
    try {
      const result = await processJob(job);
      job.dataUrl = result.dataUrl;
      job.originalDataUrl = result.originalDataUrl;
      job.result = result;
      job.status = "done";
      processed.push(result);
    } catch (err) {
      job.status = "error";
      job.error = err.message || String(err);
      console.error(`Failed to process ${job.file.name}`, err);
    }
    renderList();
  }

  for (const [fileIndex, file] of files.entries()) {
    updateProgress(`Importing file ${fileIndex + 1} of ${files.length}: ${file.name}`, fileIndex, files.length);
    if (!isPdf(file)) {
      const job = { id: jobs.length, file, label: file.name, status: "pending" };
      await processAndDisplay(job, `Processing ${file.name}`);
      updateProgress(`Processed file ${fileIndex + 1} of ${files.length}`, fileIndex + 1, files.length);
      continue;
    }

    try {
      await forEachPdfPage(file, async (sourceCanvas, pageNumber, pageCount) => {
        const job = {
          id: jobs.length,
          file,
          label: `${file.name} - page ${pageNumber}`,
          pageNumber,
          sourceCanvas,
          status: "pending",
        };
        await processAndDisplay(job, `Processing ${file.name}: page ${pageNumber} of ${pageCount}`);
        updateProgress(
          `Processed ${file.name}: page ${pageNumber} of ${pageCount}`,
          fileIndex + pageNumber / pageCount,
          files.length
        );
      });
    } catch (err) {
      jobs.push({
        id: jobs.length,
        file,
        label: file.name,
        status: "error",
        error: err.message || String(err),
        loadError: true,
      });
      renderList();
    }
  }

  processing = false;
  updateProgress(
    jobs.length
      ? `Finished: ${processed.length} of ${jobs.length} images/pages processed`
      : "No pages available to process",
    1,
    1
  );
  btnPdf.disabled = processed.length === 0;
  btnRedo.disabled = processed.length === 0;
}

// ---------------------------------------------------------------------------
// Redo & PDF
// ---------------------------------------------------------------------------

// Re-process all previously uploaded images with current settings
async function redoAll() {
  if (jobs.length === 0 || processing) return;

  processing = true;
  btnPdf.disabled = true;
  btnRedo.disabled = true;
  processed = [];
  updateProgress("Preparing pages for re-processing", null);

  const jobsToProcess = jobs.filter((job) => !job.loadError);
  let completed = 0;
  for (const [jobIndex, job] of jobsToProcess.entries()) {
    if (job.loadError) continue;
    updateProgress(`Processing ${jobIndex + 1} of ${jobsToProcess.length}: ${job.label}`, completed, jobsToProcess.length);
    job.status = "processing";
    job.dataUrl = null;
    job.error = null;
    renderList();
    try {
      const result = await processJob(job);
      job.dataUrl = result.dataUrl;
      job.originalDataUrl = result.originalDataUrl;
      job.result = result;
      job.status = "done";
      processed.push(result);
    } catch (err) {
      job.status = "error";
      job.error = err.message || String(err);
      console.error(`Redo failed for ${job.file.name}`, err);
    }
    completed += 1;
    updateProgress(`Processed ${completed} of ${jobsToProcess.length} pages`, completed, jobsToProcess.length);
    renderList();
  }

  processing = false;
  updateProgress(
    jobsToProcess.length
      ? `Finished: ${processed.length} of ${jobsToProcess.length} pages processed`
      : "No pages available to process",
    1,
    1
  );
  btnPdf.disabled = processed.length === 0;
  btnRedo.disabled = processed.length === 0;
}

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});
dropZone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");

  let files = [];
  if (e.dataTransfer.items && e.dataTransfer.items.length) {
    // Use items API so dropped folders are walked recursively
    files = await collectFilesFromItems(Array.from(e.dataTransfer.items));
  } else if (e.dataTransfer.files.length) {
    files = Array.from(e.dataTransfer.files).filter(isSupportedFile);
  }

  if (files.length) handleFiles(files);
  else if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});

dropZone.addEventListener("click", () => {
  fileInput.value = null;
  fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) handleFiles(e.target.files);
  fileInput.value = "";
});

btnRedo.addEventListener("click", redoAll);

btnPdf.addEventListener("click", async () => {
  if (processed.length === 0) return;

  btnPdf.disabled = true;
  btnPdf.textContent = "Generating PDF\u2026";

  try {
    await generatePDF({
      images: jobs.filter((job) => job.status === "done").map((job) => job.dataUrl),
      params: settings,
    });
  } catch (err) {
    console.error("PDF generation failed", err);
    alert("Failed to generate PDF: " + (err.message || String(err)));
  } finally {
    btnPdf.disabled = false;
    btnPdf.textContent = "Generate PDF";
  }
});

// ---------------------------------------------------------------------------
// Modal "Edit corners" button
// ---------------------------------------------------------------------------

modalEditCorners.addEventListener("click", async (e) => {
  e.stopPropagation();
  closeModal();

  const job = jobs.find((j) => j.id === jobIdBeingViewed);
  if (!job) return;

  const reProcessed = await reprocessJobWithCorners(job);
  if (reProcessed) {
    renderList();
    openModal(job);
  }
});

// Track which job is currently being viewed in the modal
let jobIdBeingViewed = null;

const _openModal = openModal;
openModal = (job) => {
  jobIdBeingViewed = job.id;
  _openModal(job);
};

// ---------------------------------------------------------------------------
// Initialization: wait for OpenCV, then wire up
// ---------------------------------------------------------------------------

async function init() {
  if (!window.cv?.Mat) {
    await new Promise((resolve) => {
      window.addEventListener("opencv-ready", resolve, { once: true });
    });
  }

  wireControls();
  console.log("correctified initialized.");
}

init().catch(console.error);
