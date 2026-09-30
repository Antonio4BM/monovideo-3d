/**
 * Helpers for listing finished reconstructions and selecting a GLB model.
 */

import { frameViewerOnCloud } from "./glb-framing.js";

/**
 * Fetches reconstructed and fused job lists from the API.
 *
 * @param {string} [endpoint] List URL.
 * @param {typeof fetch} [fetchImpl] Fetch implementation (injectable for tests).
 * @returns {Promise<{
 *   reconstructions: Array<{ job_id: string, glb_url: string, modified_at: string }>,
 *   fused: Array<{ job_id: string, glb_url: string, modified_at: string }>,
 * }>}
 */
export async function fetchReconstructionCatalog(
  endpoint = "/api/reconstructions",
  fetchImpl = fetch,
) {
  const response = await fetchImpl(endpoint);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const payload = await response.json();
  return {
    reconstructions: Array.isArray(payload.reconstructions) ? payload.reconstructions : [],
    fused: Array.isArray(payload.fused) ? payload.fused : [],
  };
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
