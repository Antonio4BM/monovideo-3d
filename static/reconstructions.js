import {
  createModelDialog,
  fetchReconstructionCatalog,
  renderReconstructionTable,
} from "./gallery.js";

/**
 * Loads the metadata table and wires its model actions.
 *
 * @param {Document} [root] Page document, injectable for tests.
 * @param {typeof fetch} [fetchImpl] Catalog fetch implementation.
 * @returns {Promise<void>}
 */
export async function initReconstructions(root = document, fetchImpl = fetch) {
  const statusEl = root.querySelector("#catalog-status");
  const noticeEl = root.querySelector("#catalog-notice");
  const tableEl = root.querySelector("#reconstruction-table");
  const regionEl = root.querySelector("#table-region");
  const modal = createModelDialog(root.querySelector("#model-dialog"));
  statusEl.textContent = "Loading reconstruction metadata…";
  regionEl.setAttribute("aria-busy", "true");
  try {
    const { jobs } = await fetchReconstructionCatalog("/api/reconstructions", fetchImpl);
    renderReconstructionTable(
      tableEl,
      jobs,
      (job, type, button) => {
        noticeEl.textContent = "";
        modal.open(job, type, button);
      },
      (message) => { noticeEl.textContent = message; },
    );
    statusEl.textContent = jobs.length
      ? `${jobs.length} published job${jobs.length === 1 ? "" : "s"}`
      : "No published reconstructions yet.";
  } catch (err) {
    statusEl.textContent = err instanceof Error ? err.message : "Unable to load reconstructions.";
    statusEl.dataset.state = "error";
  } finally {
    regionEl.setAttribute("aria-busy", "false");
  }
}

if (document.querySelector("#reconstruction-table")) void initReconstructions();
