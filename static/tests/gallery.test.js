import { describe, expect, it, vi } from "vitest";
import {
  fetchReconstructionCatalog,
  fetchReconstructions,
  formatModifiedAt,
  markSelectedJob,
  renderReconstructionList,
  showReconstruction,
} from "../gallery.js";

const JOB = {
  job_id: "11111111-1111-1111-1111-111111111111",
  glb_url: "/api/reconstructions/11111111-1111-1111-1111-111111111111/glb",
  modified_at: "2026-09-24T12:00:00+00:00",
};

const FUSED_JOB = {
  job_id: "22222222-2222-2222-2222-222222222222",
  glb_url: "/api/reconstructions/22222222-2222-2222-2222-222222222222/fused",
  modified_at: "2026-09-24T13:00:00+00:00",
};

describe("fetchReconstructions", () => {
  it("returns the reconstructions array from JSON", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({ reconstructions: [JOB] }),
    }));

    await expect(fetchReconstructions("/api/reconstructions", fetchImpl)).resolves.toEqual([
      JOB,
    ]);
    expect(fetchImpl).toHaveBeenCalledWith("/api/reconstructions");
  });

  it("returns an empty list when the payload has no reconstructions", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({}),
    }));

    await expect(fetchReconstructions("/api/reconstructions", fetchImpl)).resolves.toEqual([]);
  });

  it("returns reconstructed jobs from a catalog that also has fused models", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({ reconstructions: [JOB], fused: [FUSED_JOB] }),
    }));

    await expect(fetchReconstructions("/api/reconstructions", fetchImpl)).resolves.toEqual([
      JOB,
    ]);
  });

  it("throws when the response is not ok", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: false,
      text: async () => "server error",
    }));

    await expect(fetchReconstructions("/api/reconstructions", fetchImpl)).rejects.toThrow(
      "server error",
    );
  });
});

describe("fetchReconstructionCatalog", () => {
  it("returns reconstructed and fused lists", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({ reconstructions: [JOB], fused: [FUSED_JOB] }),
    }));

    await expect(fetchReconstructionCatalog("/api/reconstructions", fetchImpl)).resolves.toEqual({
      reconstructions: [JOB],
      fused: [FUSED_JOB],
    });
  });

  it("defaults missing lists to empty arrays", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({}),
    }));

    await expect(fetchReconstructionCatalog("/api/reconstructions", fetchImpl)).resolves.toEqual({
      reconstructions: [],
      fused: [],
    });
  });
});

describe("formatModifiedAt", () => {
  it("formats a valid ISO timestamp", () => {
    expect(formatModifiedAt(JOB.modified_at)).toBe(new Date(JOB.modified_at).toLocaleString());
  });

  it("returns the original string when the date is invalid", () => {
    expect(formatModifiedAt("not-a-date")).toBe("not-a-date");
  });
});

describe("renderReconstructionList", () => {
  it("renders a button per job and notifies on select", () => {
    const listEl = document.createElement("ul");
    const onSelect = vi.fn();

    renderReconstructionList(listEl, [JOB], onSelect);

    const button = listEl.querySelector("button");
    expect(button?.dataset.jobId).toBe(JOB.job_id);
    expect(button?.textContent).toContain(JOB.job_id);
    button?.click();
    expect(onSelect).toHaveBeenCalledWith(JOB);
  });

  it("renders fused jobs with their glb url", () => {
    const listEl = document.createElement("ul");
    const onSelect = vi.fn();

    renderReconstructionList(listEl, [FUSED_JOB], onSelect);

    const button = listEl.querySelector("button");
    expect(button?.dataset.jobId).toBe(FUSED_JOB.job_id);
    expect(button?.dataset.glbUrl).toBe(FUSED_JOB.glb_url);
    button?.click();
    expect(onSelect).toHaveBeenCalledWith(FUSED_JOB);
  });
});

describe("markSelectedJob", () => {
  it("marks the matching job as current", () => {
    const listEl = document.createElement("ul");
    renderReconstructionList(listEl, [JOB], () => {});

    const other = document.createElement("li");
    const otherButton = document.createElement("button");
    otherButton.dataset.jobId = "22222222-2222-2222-2222-222222222222";
    otherButton.setAttribute("aria-current", "true");
    other.append(otherButton);
    listEl.append(other);

    const selected = markSelectedJob(listEl, JOB.job_id);
    expect(selected?.getAttribute("aria-current")).toBe("true");
    expect(otherButton.hasAttribute("aria-current")).toBe(false);
  });
});

describe("showReconstruction", () => {
  it("sets the viewer source and hides the empty state", () => {
    const viewerEl = document.createElement("div");
    viewerEl.hidden = true;
    const emptyEl = document.createElement("p");

    showReconstruction(viewerEl, emptyEl, JOB.glb_url);

    expect(viewerEl.getAttribute("src")).toBe(JOB.glb_url);
    expect(viewerEl.hidden).toBe(false);
    expect(emptyEl.hidden).toBe(true);
  });
});
