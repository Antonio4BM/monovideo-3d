import { describe, expect, it, vi } from "vitest";
import {
  MAX_DURATION_SEC,
  MIN_DURATION_SEC,
  durationConstraintMessage,
  getVideoDuration,
  hideDurationModal,
  isDurationAllowed,
  isVideoFile,
  showDurationModal,
  uploadVideo,
  validateVideoFile,
} from "../uploader.js";

/**
 * @param {Partial<File>} overrides
 * @returns {File}
 */
function makeFile(overrides = {}) {
  return /** @type {File} */ ({
    name: "clip.mp4",
    type: "video/mp4",
    ...overrides,
  });
}

describe("isVideoFile", () => {
  it("accepts video MIME types", () => {
    expect(isVideoFile(makeFile({ type: "video/mp4" }))).toBe(true);
  });

  it("rejects non-video and empty values", () => {
    expect(isVideoFile(makeFile({ type: "image/png" }))).toBe(false);
    expect(isVideoFile(null)).toBe(false);
    expect(isVideoFile(undefined)).toBe(false);
  });
});

describe("isDurationAllowed", () => {
  it("allows the inclusive 20–40 second range", () => {
    expect(isDurationAllowed(20)).toBe(true);
    expect(isDurationAllowed(30)).toBe(true);
    expect(isDurationAllowed(40)).toBe(true);
  });

  it("rejects durations outside the range or non-finite values", () => {
    expect(isDurationAllowed(19.9)).toBe(false);
    expect(isDurationAllowed(40.1)).toBe(false);
    expect(isDurationAllowed(Number.NaN)).toBe(false);
  });
});

describe("durationConstraintMessage", () => {
  it("states the configured duration constraint", () => {
    expect(durationConstraintMessage()).toBe(
      `The video length should be between ${MIN_DURATION_SEC} seconds and ${MAX_DURATION_SEC} seconds.`,
    );
    expect(durationConstraintMessage(15, 45)).toBe(
      "The video length should be between 15 seconds and 45 seconds.",
    );
  });
});

describe("duration modal helpers", () => {
  it("shows and hides the modal with a warning message", () => {
    const modal = document.createElement("div");
    modal.hidden = true;
    const messageEl = document.createElement("p");

    showDurationModal(modal, messageEl, durationConstraintMessage());
    expect(modal.hidden).toBe(false);
    expect(modal.getAttribute("aria-hidden")).toBe("false");
    expect(messageEl.textContent).toContain("20 seconds and 40 seconds");

    hideDurationModal(modal);
    expect(modal.hidden).toBe(true);
    expect(modal.getAttribute("aria-hidden")).toBe("true");
  });
});

describe("validateVideoFile", () => {
  it("rejects non-video MIME types", async () => {
    const result = await validateVideoFile(makeFile({ type: "text/plain" }));
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toBe("mime");
    }
  });

  it("rejects videos outside the duration window", async () => {
    const result = await validateVideoFile(makeFile(), {
      getDuration: async () => 12,
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toBe("duration");
      expect(result.message).toBe(durationConstraintMessage());
    }
  });

  it("rejects videos when metadata cannot be read", async () => {
    const result = await validateVideoFile(makeFile(), {
      getDuration: async () => {
        throw new Error("boom");
      },
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toBe("metadata");
    }
  });

  it("accepts videos within the duration window", async () => {
    const file = makeFile();
    const result = await validateVideoFile(file, {
      getDuration: async () => 25,
    });
    expect(result).toEqual({ ok: true, file, duration: 25 });
  });
});

describe("getVideoDuration", () => {
  it("resolves duration from loaded metadata", async () => {
    const createObjectURL = vi.fn(() => "blob:mock");
    const revokeObjectURL = vi.fn();
    const originalCreate = document.createElement.bind(document);

    vi.spyOn(document, "createElement").mockImplementation((tag) => {
      if (tag !== "video") return originalCreate(tag);
      const video = originalCreate("video");
      Object.defineProperty(video, "duration", { configurable: true, get: () => 33 });
      queueMicrotask(() => {
        video.onloadedmetadata?.(new Event("loadedmetadata"));
      });
      return video;
    });

    await expect(
      getVideoDuration(makeFile(), createObjectURL, revokeObjectURL),
    ).resolves.toBe(33);
    expect(createObjectURL).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mock");

    vi.restoreAllMocks();
  });

  it("rejects when duration is invalid", async () => {
    const originalCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag) => {
      if (tag !== "video") return originalCreate(tag);
      const video = originalCreate("video");
      Object.defineProperty(video, "duration", { configurable: true, get: () => Number.NaN });
      queueMicrotask(() => {
        video.onloadedmetadata?.(new Event("loadedmetadata"));
      });
      return video;
    });

    await expect(
      getVideoDuration(makeFile(), () => "blob:x", () => {}),
    ).rejects.toThrow(/duration/i);

    vi.restoreAllMocks();
  });

  it("rejects when the video element errors", async () => {
    const originalCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag) => {
      if (tag !== "video") return originalCreate(tag);
      const video = originalCreate("video");
      queueMicrotask(() => {
        video.onerror?.(new Event("error"));
      });
      return video;
    });

    await expect(
      getVideoDuration(makeFile(), () => "blob:x", () => {}),
    ).rejects.toThrow(/metadata/i);

    vi.restoreAllMocks();
  });
});

describe("uploadVideo", () => {
  it("posts multipart data and returns JSON", async () => {
    const file = makeFile();
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({ message: "Video uploaded successfully" }),
    }));

    await expect(uploadVideo(file, "/video-upload", fetchImpl)).resolves.toEqual({
      message: "Video uploaded successfully",
    });
    expect(fetchImpl).toHaveBeenCalledWith(
      "/video-upload",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("throws when the response is not ok", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: false,
      text: async () => "server error",
    }));

    await expect(uploadVideo(makeFile(), "/video-upload", fetchImpl)).rejects.toThrow(
      "server error",
    );
  });
});
