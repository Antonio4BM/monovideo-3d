import {
  fetchReconstructionCatalog,
  markSelectedJob,
  renderReconstructionList,
  showReconstruction,
} from "./gallery.js";

const reconstructedListEl = document.querySelector("#reconstruction-list");
const fusedListEl = document.querySelector("#fused-list");
const listStatus = document.querySelector("#list-status");
const fusedListStatus = document.querySelector("#fused-list-status");
const viewerEl = document.querySelector("#model-viewer");
const emptyEl = document.querySelector("#viewer-empty");

/**
 * Loads a reconstruction into the viewer and highlights it in its list.
 *
 * @param {{ job_id: string, glb_url: string }} job Selected finished job.
 * @param {HTMLElement} sourceList List that owns the clicked job.
 * @returns {void}
 */
function selectJob(job, sourceList) {
  markSelectedJob(reconstructedListEl, "");
  markSelectedJob(fusedListEl, "");
  markSelectedJob(sourceList, job.job_id);
  showReconstruction(viewerEl, emptyEl, job.glb_url);
}

/**
 * Renders a job list and updates its status message.
 *
 * @param {HTMLElement} listEl Target list element.
 * @param {HTMLElement} statusEl Status text element.
 * @param {Array<{ job_id: string, glb_url: string, modified_at: string }>} jobs
 *   Finished reconstructions for this list.
 * @param {string} emptyMessage Status text when the list is empty.
 * @returns {void}
 */
function populateList(listEl, statusEl, jobs, emptyMessage) {
  renderReconstructionList(listEl, jobs, (job) => selectJob(job, listEl));

  if (jobs.length === 0) {
    statusEl.textContent = emptyMessage;
    return;
  }

  statusEl.textContent = `${jobs.length} finished job${jobs.length === 1 ? "" : "s"}`;
}

/**
 * Loads finished reconstructions and fused models into their lists.
 *
 * @returns {Promise<void>}
 */
async function init() {
  listStatus.textContent = "Loading…";
  fusedListStatus.textContent = "Loading…";

  try {
    const catalog = await fetchReconstructionCatalog();
    populateList(
      reconstructedListEl,
      listStatus,
      catalog.reconstructions,
      "No finished reconstructions yet.",
    );
    populateList(
      fusedListEl,
      fusedListStatus,
      catalog.fused,
      "No finished fused models yet.",
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unable to load reconstructions.";
    listStatus.textContent = message;
    fusedListStatus.textContent = message;
  }
}

void init();
