import {
  durationConstraintMessage,
  hideDurationModal,
  showDurationModal,
  uploadVideo,
  validateVideoFile,
} from "./uploader.js";

const dropZone = document.querySelector("#drop-zone");
const fileInput = document.querySelector("#file-input");
const fileName = document.querySelector("#file-name");
const uploadBtn = document.querySelector("#upload-btn");
const statusElement = document.querySelector("#status");
const durationModal = document.querySelector("#duration-modal");
const durationModalMessage = document.querySelector("#duration-modal-message");
const durationModalClose = document.querySelector("#duration-modal-close");

/** @type {File | null} */
let selectedFile = null;

/**
 * Updates the selected file UI and upload button state.
 *
 * @param {File | null} file Accepted file, or null to clear.
 * @returns {void}
 */
function setFile(file) {
  selectedFile = file;
  fileName.textContent = file ? file.name : "No file selected";
  uploadBtn.disabled = !file;
}

/**
 * Clears the file input so the same path can be re-selected later.
 *
 * @returns {void}
 */
function resetFileInput() {
  fileInput.value = "";
}

/**
 * Opens the duration constraint warning modal.
 *
 * @param {string} message Warning text.
 * @returns {void}
 */
function openDurationWarning(message) {
  showDurationModal(durationModal, durationModalMessage, message);
}

/**
 * Closes the duration constraint warning modal.
 *
 * @returns {void}
 */
function closeDurationWarning() {
  hideDurationModal(durationModal);
}

/**
 * Validates a user-picked file and either accepts it or shows a warning.
 *
 * @param {File | undefined | null} file Candidate file.
 * @returns {Promise<void>}
 */
async function handleFileCandidate(file) {
  statusElement.textContent = "";

  if (!file) {
    setFile(null);
    return;
  }

  statusElement.textContent = "Checking video…";
  uploadBtn.disabled = true;

  const result = await validateVideoFile(file);

  if (!result.ok) {
    setFile(null);
    resetFileInput();
    statusElement.textContent = "";

    if (result.reason === "duration") {
      openDurationWarning(result.message);
      return;
    }

    statusElement.textContent = result.message;
    return;
  }

  setFile(result.file);
  statusElement.textContent = `Ready (${Math.round(result.duration)}s)`;
}

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  void handleFileCandidate(fileInput.files?.[0] ?? null);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  void handleFileCandidate(event.dataTransfer?.files[0] ?? null);
});

durationModalClose.addEventListener("click", closeDurationWarning);

durationModal.querySelector("[data-close-modal]")?.addEventListener("click", closeDurationWarning);

uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  statusElement.textContent = "Uploading…";
  uploadBtn.disabled = true;

  try {
    const result = await uploadVideo(selectedFile);
    statusElement.textContent = result.message ?? "Upload complete";
  } catch (err) {
    statusElement.textContent = err instanceof Error ? err.message : "Upload failed";
  } finally {
    uploadBtn.disabled = !selectedFile;
  }
});

// Keep the constraint message in sync with configured limits.
durationModalMessage.textContent = durationConstraintMessage();
