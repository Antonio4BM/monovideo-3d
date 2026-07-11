/**
 * Client-side video upload helpers: MIME check, duration validation, and POST.
 */

/** @type {number} Minimum allowed video duration in seconds. */
export const MIN_DURATION_SEC = Number(globalThis.__UPLOAD_CONFIG__?.minDurationSec ?? 20);

/** @type {number} Maximum allowed video duration in seconds. */
export const MAX_DURATION_SEC = Number(globalThis.__UPLOAD_CONFIG__?.maxDurationSec ?? 40);

/**
 * Returns whether the file looks like a video by MIME type.
 *
 * @param {File | null | undefined} file Candidate file.
 * @returns {boolean} True when `file.type` starts with `video/`.
 */
export function isVideoFile(file) {
  return Boolean(file && typeof file.type === "string" && file.type.startsWith("video/"));
}

/**
 * Reads media duration for a video file via a temporary video element.
 *
 * @param {File} file Video file to inspect.
 * @param {(file: File) => string} [createObjectURL] URL factory (injectable for tests).
 * @param {(url: string) => void} [revokeObjectURL] URL cleanup (injectable for tests).
 * @returns {Promise<number>} Duration in seconds.
 */
export function getVideoDuration(
  file,
  createObjectURL = (f) => URL.createObjectURL(f),
  revokeObjectURL = (url) => URL.revokeObjectURL(url),
) {
  return new Promise((resolve, reject) => {
    const url = createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";

    const cleanup = () => {
      video.removeAttribute("src");
      revokeObjectURL(url);
    };

    video.onloadedmetadata = () => {
      const duration = video.duration;
      cleanup();
      if (!Number.isFinite(duration) || duration <= 0) {
        reject(new Error("Unable to determine video duration."));
        return;
      }
      resolve(duration);
    };

    video.onerror = () => {
      cleanup();
      reject(new Error("Unable to read video metadata."));
    };

    video.src = url;
  });
}

/**
 * Checks whether a duration falls within the allowed range.
 *
 * @param {number} durationSec Duration in seconds.
 * @param {number} [minSec] Inclusive minimum (defaults to env/config).
 * @param {number} [maxSec] Inclusive maximum (defaults to env/config).
 * @returns {boolean} True when duration is within bounds.
 */
export function isDurationAllowed(
  durationSec,
  minSec = MIN_DURATION_SEC,
  maxSec = MAX_DURATION_SEC,
) {
  return Number.isFinite(durationSec) && durationSec >= minSec && durationSec <= maxSec;
}

/**
 * Builds the warning copy for an out-of-range video.
 *
 * @param {number} [minSec] Inclusive minimum seconds.
 * @param {number} [maxSec] Inclusive maximum seconds.
 * @returns {string} User-facing warning message.
 */
export function durationConstraintMessage(
  minSec = MIN_DURATION_SEC,
  maxSec = MAX_DURATION_SEC,
) {
  return `The video length should be between ${minSec} seconds and ${maxSec} seconds.`;
}

/**
 * Shows the duration warning modal and updates its message.
 *
 * @param {HTMLElement} modal Modal root element.
 * @param {HTMLElement | null} messageEl Element that holds the warning text.
 * @param {string} message Warning message to display.
 * @returns {void}
 */
export function showDurationModal(modal, messageEl, message) {
  if (messageEl) {
    messageEl.textContent = message;
  }
  modal.hidden = false;
  modal.setAttribute("aria-hidden", "false");
}

/**
 * Hides the duration warning modal.
 *
 * @param {HTMLElement} modal Modal root element.
 * @returns {void}
 */
export function hideDurationModal(modal) {
  modal.hidden = true;
  modal.setAttribute("aria-hidden", "true");
}

/**
 * Validates a selected file and returns it only when MIME and duration pass.
 *
 * @param {File} file Candidate video file.
 * @param {{
 *   getDuration?: (file: File) => Promise<number>,
 *   minSec?: number,
 *   maxSec?: number,
 * }} [options] Optional overrides for tests and config.
 * @returns {Promise<{ ok: true, file: File, duration: number } | { ok: false, reason: "mime" | "duration" | "metadata", message: string }>}
 */
export async function validateVideoFile(file, options = {}) {
  const {
    getDuration = getVideoDuration,
    minSec = MIN_DURATION_SEC,
    maxSec = MAX_DURATION_SEC,
  } = options;

  if (!isVideoFile(file)) {
    return {
      ok: false,
      reason: "mime",
      message: "Please choose a video file.",
    };
  }

  let duration;
  try {
    duration = await getDuration(file);
  } catch {
    return {
      ok: false,
      reason: "metadata",
      message: "Unable to read video duration. Please try another file.",
    };
  }

  if (!isDurationAllowed(duration, minSec, maxSec)) {
    return {
      ok: false,
      reason: "duration",
      message: durationConstraintMessage(minSec, maxSec),
    };
  }

  return { ok: true, file, duration };
}

/**
 * Uploads a validated video to the app endpoint.
 *
 * @param {File} file Video file to upload.
 * @param {string} [endpoint] Upload URL.
 * @param {typeof fetch} [fetchImpl] Fetch implementation (injectable for tests).
 * @returns {Promise<{ message: string }>} Parsed JSON response body.
 */
export async function uploadVideo(file, endpoint = "/video-upload", fetchImpl = fetch) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetchImpl(endpoint, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return /** @type {{ message: string }} */ (await response.json());
}
