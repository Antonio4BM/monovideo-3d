import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PAGE from "../reconstructions.html?raw";
import {
  CATALOG_TIMEOUT_MS,
  createModelDialog,
  DENSE_COMING_SOON,
  fetchReconstructionCatalog,
  formatMetadataValue,
  METADATA_COLUMNS,
  renderReconstructionTable,
} from "../gallery.js";
import { initReconstructions } from "../reconstructions.js";

const JOB = {
  job_id: "11111111-1111-1111-1111-111111111111",
  feature_extractor_algorithm: "SIFT",
  strides: 20,
  completed_at: "2026-11-19T12:30:00Z",
  reconstruction_duration_seconds: 40.25,
  database_snapshot: "artifacts/database.db",
  feature_archive: null,
  summary_path: "artifacts/summary.json",
  image_count: 5,
  total_keypoints: 2000,
  total_matches: 0,
  total_descriptors: 2000,
  median_triangulation_angle_degrees: 0,
  sparse_glb_url: "/api/reconstructions/11111111-1111-1111-1111-111111111111/glb",
  dense_glb_url: null,
};

/** Provide jsdom's missing native dialog methods for controller tests. */
function preparePage() {
  document.body.innerHTML = PAGE;
  const dialog = document.querySelector("dialog");
  dialog.showModal = vi.fn(() => {
    dialog.open = true;
    document.querySelector("#close-model").focus();
  });
  dialog.close = vi.fn(() => {
    dialog.open = false;
    dialog.dispatchEvent(new Event("close"));
  });
  return dialog;
}

beforeEach(() => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: false });
});
afterEach(() => {
  document.querySelector("dialog[open]")?.close();
  document.body.replaceChildren();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("metadata catalog", () => {
  it.each(["headers", "body"])("aborts when the response stalls at %s", async (stage) => {
    vi.useFakeTimers();
    let requestSignal;
    const fetchImpl = vi.fn((_endpoint, { signal }) => {
      requestSignal = signal;
      const stalled = new Promise((_resolve, reject) => {
        signal.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"));
        });
      });
      return stage === "headers"
        ? stalled
        : Promise.resolve({ ok: true, json: () => stalled });
    });
    const pending = fetchReconstructionCatalog("/catalog", fetchImpl);
    const assertion = expect(pending).rejects.toThrow("timed out");
    await vi.advanceTimersByTimeAsync(CATALOG_TIMEOUT_MS - 1);
    expect(requestSignal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await assertion;
    expect(requestSignal.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it.each([true, false])("clears the timer after a completed response (ok=%s)", async (ok) => {
    vi.useFakeTimers();
    let requestSignal;
    const pending = fetchReconstructionCatalog("/catalog", async (_endpoint, { signal }) => {
      requestSignal = signal;
      return { ok, json: async () => ({ jobs: [JOB] }), text: async () => "Server error" };
    });
    if (ok) await expect(pending).resolves.toMatchObject({ jobs: [JOB] });
    else await expect(pending).rejects.toThrow("Server error");
    expect(vi.getTimerCount()).toBe(0);
    await vi.advanceTimersByTimeAsync(CATALOG_TIMEOUT_MS);
    expect(requestSignal.aborted).toBe(false);
  });

  it("keeps metadata and defaults invalid legacy lists to empty arrays", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({ jobs: [JOB], reconstructions: null, fused: {} }),
    }));
    await expect(fetchReconstructionCatalog("/catalog", fetchImpl)).resolves.toEqual({
      jobs: [JOB], reconstructions: [], fused: [],
    });
  });

  it("extracts a readable FastAPI error message", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: false,
      text: async () => JSON.stringify({ detail: "DYNAMODB_TABLE_NAME must be configured." }),
    }));
    await expect(fetchReconstructionCatalog("/catalog", fetchImpl)).rejects.toThrow(
      "DYNAMODB_TABLE_NAME must be configured.",
    );
  });

  it("provides a fallback for an empty server error", async () => {
    const fetchImpl = vi.fn(async () => ({ ok: false, text: async () => "" }));
    await expect(fetchReconstructionCatalog("/catalog", fetchImpl)).rejects.toThrow(
      "Unable to load reconstructions.",
    );
  });
});

describe("formatMetadataValue", () => {
  it.each([null, undefined, ""])("uses a dash for %s", (value) => {
    expect(formatMetadataValue("image_count", value)).toBe("—");
  });
  it("preserves zero and formats counts, durations, and dates", () => {
    expect(formatMetadataValue("total_matches", 0)).toBe("0");
    expect(formatMetadataValue("total_keypoints", 2000)).toBe((2000).toLocaleString());
    expect(formatMetadataValue("reconstruction_duration_seconds", 1.25)).toBe(
      (1.25).toLocaleString(undefined, { maximumFractionDigits: 3 }),
    );
    expect(formatMetadataValue("completed_at", JOB.completed_at)).toBe(
      new Date(JOB.completed_at).toLocaleString(),
    );
    expect(formatMetadataValue("completed_at", "invalid")).toBe("invalid");
  });
});

describe("reconstruction table", () => {
  it("renders metadata in order and omits local artifact paths from older records", () => {
    const table = document.createElement("table");
    renderReconstructionTable(table, [JOB], vi.fn(), vi.fn());
    const headers = [...table.querySelectorAll("th")];
    expect(headers.map((cell) => cell.textContent)).toEqual([
      ...METADATA_COLUMNS.map(([, title]) => title), "Sparse", "Dense",
    ]);
    expect(headers.every((cell) => cell.scope === "col")).toBe(true);
    const cells = [...table.querySelectorAll("tbody td")];
    expect(cells.slice(0, METADATA_COLUMNS.length).map((cell) => cell.textContent)).toEqual(
      METADATA_COLUMNS.map(([field]) => formatMetadataValue(field, JOB[field])),
    );
    expect(table.querySelectorAll("a")).toHaveLength(0);
    expect(cells).toHaveLength(METADATA_COLUMNS.length + 2);
    expect(table.textContent).not.toContain("Database snapshot");
    expect(table.textContent).not.toContain("Feature archive");
    expect(table.textContent).not.toContain("Summary path");
    expect(table.textContent).not.toContain("artifacts/");
  });

  it("renders untrusted metadata as text and replaces old rows", () => {
    const table = document.createElement("table");
    renderReconstructionTable(table, [JOB, JOB], vi.fn(), vi.fn());
    renderReconstructionTable(table, [{ ...JOB, feature_extractor_algorithm: '<img src=x onerror="alert(1)">' }], vi.fn(), vi.fn());
    expect(table.querySelectorAll("tbody tr")).toHaveLength(1);
    expect(table.querySelector("img")).toBeNull();
    expect(table.textContent).toContain('<img src=x onerror="alert(1)">');
  });

  it.each([
    [null, null, true, true],
    ["/sparse.glb", null, false, false],
    [null, "/dense.glb", true, false],
    ["/sparse.glb", "/dense.glb", false, false],
  ])("handles sparse=%s dense=%s", (sparse, dense, sparseDisabled, denseDisabled) => {
    const table = document.createElement("table");
    const job = { ...JOB, sparse_glb_url: sparse, dense_glb_url: dense };
    const onView = vi.fn();
    const onNotice = vi.fn();
    renderReconstructionTable(table, [job], onView, onNotice);
    const [sparseButton, denseButton] = table.querySelectorAll("button");
    expect(sparseButton.disabled).toBe(sparseDisabled);
    expect(denseButton.disabled).toBe(denseDisabled);
    sparseButton.click();
    denseButton.click();
    if (sparse) expect(onView).toHaveBeenCalledWith(job, "sparse", sparseButton);
    if (dense) expect(onView).toHaveBeenCalledWith(job, "dense", denseButton);
    if (sparse && !dense) {
      expect(denseButton.textContent).toBe("Make dense");
      expect(onNotice).toHaveBeenCalledWith(DENSE_COMING_SOON);
    } else expect(onNotice).not.toHaveBeenCalled();
    expect(onView).toHaveBeenCalledTimes(Number(Boolean(sparse)) + Number(Boolean(dense)));
  });
});

describe("model dialog", () => {
  it("opens the requested model, reports loading and success, then releases it and restores focus", () => {
    const dialog = preparePage();
    const modal = createModelDialog(dialog);
    const opener = document.createElement("button");
    document.body.append(opener);
    modal.open(JOB, "sparse", opener);
    const viewer = dialog.querySelector("model-viewer");
    const status = document.querySelector("#model-status");
    expect(dialog.open).toBe(true);
    expect(document.activeElement.id).toBe("close-model");
    expect(viewer.getAttribute("src")).toBe(JOB.sparse_glb_url);
    expect(document.querySelector("#model-title").textContent).toContain(`Sparse reconstruction · ${JOB.job_id}`);
    expect(status.dataset.state).toBe("loading");
    expect(viewer.getAttribute("orientation")).toBe("180deg 30deg 40deg");
    viewer.dispatchEvent(new Event("load"));
    expect(status.dataset.state).toBe("loaded");
    document.querySelector("#close-model").click();
    expect(dialog.open).toBe(false);
    expect(viewer.hasAttribute("src")).toBe(false);
    expect(dialog.querySelector("model-viewer")).toBeNull();
    expect(document.activeElement).toBe(opener);
  });

  it("shows model errors and ignores events from a replaced or closed viewer", () => {
    const dialog = preparePage();
    const modal = createModelDialog(dialog);
    const opener = document.createElement("button");
    document.body.append(opener);
    modal.open(JOB, "sparse", opener);
    const oldViewer = dialog.querySelector("model-viewer");
    const oldSignal = fetch.mock.calls.at(-1)[1].signal;
    modal.open({ ...JOB, dense_glb_url: "/dense.glb" }, "dense", opener);
    const viewer = dialog.querySelector("model-viewer");
    const status = document.querySelector("#model-status");
    expect(oldSignal.aborted).toBe(true);
    oldViewer.dispatchEvent(new Event("load"));
    oldViewer.dispatchEvent(new Event("error"));
    expect(status.dataset.state).toBe("loading");
    expect(viewer.getAttribute("src")).toBe("/dense.glb");
    expect(document.querySelector("#model-title").textContent).toContain("Dense reconstruction");
    viewer.dispatchEvent(new Event("error"));
    expect(status.textContent).toContain("Unable to load");
    expect(status.dataset.state).toBe("error");
    dialog.close();
    viewer.dispatchEvent(new Event("load"));
    expect(status.textContent).toBe("");
  });

  it("does not open unavailable models", () => {
    const dialog = preparePage();
    createModelDialog(dialog).open(JOB, "dense", document.createElement("button"));
    expect(dialog.showModal).not.toHaveBeenCalled();
  });
});

describe("reconstruction page", () => {
  it("replaces the loading state with a timeout message when the API stalls", async () => {
    vi.useFakeTimers();
    preparePage();
    const fetchImpl = (_endpoint, { signal }) => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => {
        reject(new DOMException("Aborted", "AbortError"));
      });
    });
    const pending = initReconstructions(document, fetchImpl);
    await vi.advanceTimersByTimeAsync(CATALOG_TIMEOUT_MS);
    await pending;
    const status = document.querySelector("#catalog-status");
    expect(status.textContent).toContain("timed out");
    expect(status.textContent).toContain("refresh the page");
    expect(status.dataset.state).toBe("error");
    expect(document.querySelector("#table-region").getAttribute("aria-busy")).toBe("false");
  });

  it("shows loading, renders published jobs, and wires the placeholder and viewer actions", async () => {
    const dialog = preparePage();
    let resolveCatalog;
    const response = new Promise((resolve) => { resolveCatalog = resolve; });
    const pending = initReconstructions(document, () => response);
    expect(document.querySelector("#catalog-status").textContent).toContain("Loading");
    expect(document.querySelector("#table-region").getAttribute("aria-busy")).toBe("true");
    resolveCatalog({ ok: true, json: async () => ({ jobs: [JOB] }) });
    await pending;
    expect(document.querySelector("#catalog-status").textContent).toBe("1 published job");
    expect(document.querySelector("#table-region").getAttribute("aria-busy")).toBe("false");
    const [sparse, dense] = document.querySelectorAll("tbody button");
    dense.click();
    expect(document.querySelector("#catalog-notice").textContent).toBe(DENSE_COMING_SOON);
    expect(dialog.open).toBe(false);
    sparse.click();
    expect(dialog.open).toBe(true);
    expect(document.querySelector("#catalog-notice").textContent).toBe("");
  });

  it.each([0, 2])("handles a catalog with %s jobs", async (count) => {
    preparePage();
    await initReconstructions(document, async () => ({
      ok: true, json: async () => ({ jobs: Array(count).fill(JOB) }),
    }));
    expect(document.querySelector("#catalog-status").textContent).toBe(
      count ? "2 published jobs" : "No published reconstructions yet.",
    );
  });

  it("shows an actionable catalog error distinct from an empty state", async () => {
    preparePage();
    await initReconstructions(document, async () => ({
      ok: false, text: async () => '{"detail":"DynamoDB is unavailable."}',
    }));
    expect(document.querySelector("#catalog-status").textContent).toBe("DynamoDB is unavailable.");
    expect(document.querySelector("#catalog-status").dataset.state).toBe("error");
    expect(document.querySelector("#table-region").getAttribute("aria-busy")).toBe("false");
  });
});
