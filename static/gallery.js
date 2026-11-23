/**
 * Helpers for listing finished reconstructions and selecting a GLB model.
 */

import { frameViewerOnCloud } from "./glb-framing.js";

/**
 * @typedef {Object} ReconstructionJob
 * @property {string} job_id Job identifier.
 * @property {string|null} feature_extractor_algorithm Feature algorithm.
 * @property {number|null} strides Frame extraction interval.
 * @property {string|null} completed_at ISO-8601 completion timestamp.
 * @property {number|null} reconstruction_duration_seconds Pipeline duration.
 * @property {number|null} image_count Number of images.
 * @property {number|null} total_keypoints Number of keypoints.
 * @property {number|null} total_matches Number of matches.
 * @property {number|null} total_descriptors Number of descriptors.
 * @property {number|null} median_triangulation_angle_degrees Median angle.
 * @property {string|null} sparse_glb_url Available reconstructed GLB URL.
 * @property {string|null} dense_glb_url Available fused GLB URL.
 */

/** Metadata columns in summary order; model actions follow these columns. */
export const METADATA_COLUMNS = [
  ["job_id", "Job ID"],
  ["feature_extractor_algorithm", "Feature algorithm"],
  ["strides", "Strides"],
  ["completed_at", "Completed at"],
  ["reconstruction_duration_seconds", "Duration (seconds)"],
  ["image_count", "Images"],
  ["total_keypoints", "Total keypoints"],
  ["total_matches", "Total matches"],
  ["total_descriptors", "Total descriptors"],
  ["median_triangulation_angle_degrees", "Median triangulation angle (degrees)"],
];

export const DENSE_COMING_SOON =
  "Dense reconstruction from sparse is not implemented yet.";

export const CATALOG_TIMEOUT_MS = 10000;

/**
 * Fetches the catalog, allowing 15 seconds for the full response body.
 *
 * @param {string} [endpoint] List URL.
 * @param {typeof fetch} [fetchImpl] Fetch implementation (injectable for tests).
 * @returns {Promise<{
 *   jobs: ReconstructionJob[],
 *   reconstructions: Array<{ job_id: string, glb_url: string, modified_at: string }>,
 *   fused: Array<{ job_id: string, glb_url: string, modified_at: string }>,
 * }>}
 */
export async function fetchReconstructionCatalog(
  endpoint = "/api/reconstructions",
  fetchImpl = fetch,
) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), CATALOG_TIMEOUT_MS);
  try {
    const response = await fetchImpl(endpoint, { signal: controller.signal });
    if (!response.ok) {
      const message = await response.text();
      let detail;
      try {
        detail = JSON.parse(message).detail;
      } catch {
        // Non-JSON errors from proxies still provide a useful message.
      }
      throw new Error(
        typeof detail === "string" ? detail : message || "Unable to load reconstructions.",
      );
    }
    const payload = await response.json();
    return {
      jobs: Array.isArray(payload.jobs) ? payload.jobs : [],
      reconstructions: Array.isArray(payload.reconstructions) ? payload.reconstructions : [],
      fused: Array.isArray(payload.fused) ? payload.fused : [],
    };
  } catch (err) {
    if (controller.signal.aborted) {
      throw new Error("Loading reconstruction metadata timed out. Please refresh the page to try again.");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Fetches the finished reconstructions list from the API.
 *
 * @param {string} [endpoint] List URL.
 * @param {typeof fetch} [fetchImpl] Fetch implementation (injectable for tests).
 * @returns {Promise<Array<{ job_id: string, glb_url: string, modified_at: string }>>}
 */
export async function fetchReconstructions(
  endpoint = "/api/reconstructions",
  fetchImpl = fetch,
) {
  const catalog = await fetchReconstructionCatalog(endpoint, fetchImpl);
  return catalog.reconstructions;
}

/**
 * Formats a reconstruction timestamp for the job list.
 *
 * @param {string} isoDate ISO-8601 timestamp from the API.
 * @returns {string} Locale date/time or the original string if invalid.
 */
export function formatModifiedAt(isoDate) {
  const date = new Date(isoDate);
  if (Number.isNaN(date.getTime())) {
    return isoDate;
  }
  return date.toLocaleString();
}

/**
 * Renders selectable reconstruction jobs into a list element.
 *
 * @param {HTMLElement} listEl Target `<ul>` or `<ol>`.
 * @param {Array<{ job_id: string, glb_url: string, modified_at: string }>} jobs
 *   Finished reconstructions.
 * @param {(job: { job_id: string, glb_url: string, modified_at: string }) => void} onSelect
 *   Called when a job button is clicked.
 * @returns {void}
 */
export function renderReconstructionList(listEl, jobs, onSelect) {
  listEl.replaceChildren();

  for (const job of jobs) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.jobId = job.job_id;
    button.dataset.glbUrl = job.glb_url;

    const idEl = document.createElement("span");
    idEl.className = "job-id";
    idEl.textContent = job.job_id;

    const dateEl = document.createElement("span");
    dateEl.className = "job-date";
    dateEl.textContent = formatModifiedAt(job.modified_at);

    button.append(idEl, dateEl);
    button.addEventListener("click", () => onSelect(job));
    item.append(button);
    listEl.append(item);
  }
}

/**
 * Marks the selected job in the list and returns its button.
 *
 * @param {HTMLElement} listEl List that holds job buttons.
 * @param {string} jobId Selected job UUID.
 * @returns {HTMLButtonElement | null} The selected button, if present.
 */
export function markSelectedJob(listEl, jobId) {
  const buttons = listEl.querySelectorAll("button[data-job-id]");
  /** @type {HTMLButtonElement | null} */
  let selected = null;

  buttons.forEach((button) => {
    const isCurrent = button.dataset.jobId === jobId;
    if (isCurrent) {
      button.setAttribute("aria-current", "true");
      selected = /** @type {HTMLButtonElement} */ (button);
    } else {
      button.removeAttribute("aria-current");
    }
  });

  return selected;
}

/**
 * Shows a GLB in the model viewer and hides the empty-state message.
 *
 * @param {HTMLElement} viewerEl `<model-viewer>` element.
 * @param {HTMLElement | null} emptyEl Empty-state message element.
 * @param {string} glbUrl URL of the reconstructed GLB.
 * @returns {void}
 */
export function showReconstruction(viewerEl, emptyEl, glbUrl) {
  viewerEl.setAttribute("src", glbUrl);
  viewerEl.hidden = false;
  if (emptyEl) {
    emptyEl.hidden = true;
  }
  void frameViewerOnCloud(viewerEl, glbUrl);
}

/**
 * Formats a metadata value without treating zero as missing.
 *
 * @param {string} field Metadata field name.
 * @param {unknown} value Value returned by the API.
 * @returns {string} Plain-text cell value.
 */
export function formatMetadataValue(field, value) {
  if (value === null || value === undefined || value === "") return "—";
  if (field === "completed_at") return formatModifiedAt(String(value));
  if (typeof value === "number") {
    return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  }
  return String(value);
}

/**
 * Renders one row per published job, including separate model actions.
 *
 * @param {HTMLTableElement} tableEl Metadata table.
 * @param {ReconstructionJob[]} jobs All published jobs.
 * @param {(job: ReconstructionJob, type: string, button: HTMLButtonElement) => void} onView
 *   Opens an available sparse or dense model.
 * @param {(message: string) => void} onNotice Displays the dense placeholder.
 * @returns {void}
 */
export function renderReconstructionTable(tableEl, jobs, onView, onNotice) {
  const head = document.createElement("thead");
  const header = head.insertRow();
  const labels = [...METADATA_COLUMNS.map(([, title]) => title), "Sparse", "Dense"];
  for (const label of labels) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    header.append(cell);
  }
  const body = document.createElement("tbody");
  for (const job of jobs) {
    const row = body.insertRow();
    for (const [field] of METADATA_COLUMNS) {
      const cell = row.insertCell();
      cell.textContent = formatMetadataValue(field, job[field]);
      if (field === "job_id") {
        cell.className = "metadata-path";
      }
    }
    for (const type of ["sparse", "dense"]) {
      const url = type === "sparse" ? job.sparse_glb_url : job.dense_glb_url;
      const canGenerateLater = type === "dense" && !url && Boolean(job.sparse_glb_url);
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = canGenerateLater ? "Make dense" : "View 3D";
      button.disabled = !url && !canGenerateLater;
      const action = canGenerateLater ? "Make" : "View";
      button.setAttribute("aria-label", `${action} ${type} reconstruction for ${job.job_id}`);
      button.title = button.disabled ? `No ${type} model is available.` : "";
      button.addEventListener("click", () => {
        if (canGenerateLater) onNotice(DENSE_COMING_SOON);
        else onView(job, type, button);
      });
      row.insertCell().append(button);
    }
  }
  tableEl.replaceChildren(head, body);
}

/**
 * Creates a modal controller with an isolated viewer for each opened model.
 *
 * @param {HTMLDialogElement} dialogEl Dialog containing a model template.
 * @returns {{open: (job: ReconstructionJob, type: string, button: HTMLButtonElement) => void}}
 *   Controller for opening the selected model.
 */
export function createModelDialog(dialogEl) {
  const titleEl = dialogEl.querySelector("#model-title");
  const statusEl = dialogEl.querySelector("#model-status");
  const hostEl = dialogEl.querySelector(".viewer-container");
  const templateEl = dialogEl.querySelector("template");
  let activeViewer = null;
  let request = null;
  let opener = null;

  /** Release the previous viewer, pending fetch, and event handlers. */
  function resetViewer() {
    request?.abort();
    request = null;
    activeViewer?.removeAttribute("src");
    activeViewer = null;
    hostEl.replaceChildren();
  }

  dialogEl.querySelector("#close-model").addEventListener("click", () => dialogEl.close());
  dialogEl.addEventListener("close", () => {
    resetViewer();
    statusEl.textContent = "";
    opener?.focus({ preventScroll: true });
    opener = null;
  });

  return {
    /** Open an available model and track its loading lifecycle. */
    open(job, type, button) {
      const glbUrl = type === "sparse" ? job.sparse_glb_url : job.dense_glb_url;
      if (!glbUrl) return;
      resetViewer();
      opener = button;
      const label = type === "sparse" ? "Sparse" : "Dense";
      titleEl.textContent = `${label} reconstruction · ${job.job_id}`;
      statusEl.textContent = "Loading 3D model…";
      statusEl.dataset.state = "loading";
      hostEl.append(templateEl.content.cloneNode(true));
      const viewer = hostEl.querySelector("model-viewer");
      activeViewer = viewer;
      request = new AbortController();
      const { signal } = request;
      viewer.setAttribute("alt", `${label} reconstruction for job ${job.job_id}`);
      viewer.addEventListener("load", () => {
        if (activeViewer !== viewer || !dialogEl.open) return;
        statusEl.textContent = "Model loaded. Drag to orbit; scroll or pinch to zoom.";
        statusEl.dataset.state = "loaded";
      }, { signal });
      viewer.addEventListener("error", () => {
        if (activeViewer !== viewer || !dialogEl.open) return;
        statusEl.textContent = "Unable to load this 3D model. Close the preview and try again.";
        statusEl.dataset.state = "error";
        request.abort();
      }, { signal });
      if (!dialogEl.open) dialogEl.showModal();
      viewer.setAttribute("src", glbUrl);
      void frameViewerOnCloud(viewer, glbUrl, fetch, signal);
    },
  };
}
