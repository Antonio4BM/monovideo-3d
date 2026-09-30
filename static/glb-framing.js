/**
 * Parse GLB position data and frame a model-viewer orbit on a robust cloud center.
 */

const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;
const FLOAT = 5126;
const VEC3_BYTES = 12;

/**
 * Reads a little-endian uint32 from a data view.
 *
 * @param {DataView} view Source view.
 * @param {number} offset Byte offset.
 * @returns {number} Unsigned 32-bit value.
 */
function u32(view, offset) {
  return view.getUint32(offset, true);
}

/**
 * Multiplies two column-major 4x4 matrices.
 *
 * @param {number[]} a Left matrix.
 * @param {number[]} b Right matrix.
 * @returns {number[]} Product matrix (16 elements).
 */
function multiplyMat4(a, b) {
  const out = new Array(16).fill(0);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[col * 4 + row] =
        a[row] * b[col * 4] +
        a[row + 4] * b[col * 4 + 1] +
        a[row + 8] * b[col * 4 + 2] +
        a[row + 12] * b[col * 4 + 3];
    }
  }
  return out;
}

/**
 * Builds a column-major 4x4 matrix from a glTF node TRS or matrix.
 *
 * @param {{ matrix?: number[], translation?: number[], rotation?: number[], scale?: number[] }} node
 *   glTF node.
 * @returns {number[]} Column-major 4x4 matrix.
 */
export function nodeToMatrix(node) {
  if (Array.isArray(node.matrix) && node.matrix.length === 16) {
    return node.matrix.map(Number);
  }

  const t = node.translation ?? [0, 0, 0];
  const r = node.rotation ?? [0, 0, 0, 1];
  const s = node.scale ?? [1, 1, 1];
  const [qx, qy, qz, qw] = r;

  const xx = qx * qx;
  const yy = qy * qy;
  const zz = qz * qz;
  const xy = qx * qy;
  const xz = qx * qz;
  const yz = qy * qz;
  const wx = qw * qx;
  const wy = qw * qy;
  const wz = qw * qz;

  return [
    (1 - 2 * (yy + zz)) * s[0],
    2 * (xy + wz) * s[0],
    2 * (xz - wy) * s[0],
    0,
    2 * (xy - wz) * s[1],
    (1 - 2 * (xx + zz)) * s[1],
    2 * (yz + wx) * s[1],
    0,
    2 * (xz + wy) * s[2],
    2 * (yz - wx) * s[2],
    (1 - 2 * (xx + yy)) * s[2],
    0,
    t[0],
    t[1],
    t[2],
    1,
  ];
}

/**
 * Transforms vec3 positions by a column-major 4x4 matrix.
 *
 * @param {Float32Array} positions Interleaved xyz values.
 * @param {number[]} matrix Column-major 4x4.
 * @returns {Float32Array} Transformed positions.
 */
export function transformPositions(positions, matrix) {
  const out = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    const w = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
    const iw = w !== 0 ? 1 / w : 1;
    out[i] = (matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) * iw;
    out[i + 1] = (matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) * iw;
    out[i + 2] = (matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]) * iw;
  }
  return out;
}

/**
 * Reads FLOAT VEC3 values from a glTF accessor in a BIN chunk.
 *
 * @param {object} json glTF JSON document.
 * @param {Uint8Array} bin BIN chunk bytes.
 * @param {number} accessorIndex Accessor index.
 * @returns {Float32Array} Interleaved xyz positions.
 */
export function readAccessorVec3(json, bin, accessorIndex) {
  const accessor = json.accessors?.[accessorIndex];
  if (!accessor || accessor.componentType !== FLOAT || accessor.type !== "VEC3") {
    return new Float32Array(0);
  }
  const view = json.bufferViews?.[accessor.bufferView];
  if (!view) {
    return new Float32Array(0);
  }
  const offset = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const stride = view.byteStride || VEC3_BYTES;
  const count = accessor.count || 0;
  const out = new Float32Array(count * 3);
  const dataView = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  for (let i = 0; i < count; i += 1) {
    const byteOffset = offset + i * stride;
    out[i * 3] = dataView.getFloat32(byteOffset, true);
    out[i * 3 + 1] = dataView.getFloat32(byteOffset + 4, true);
    out[i * 3 + 2] = dataView.getFloat32(byteOffset + 8, true);
  }
  return out;
}

/**
 * Splits a GLB ArrayBuffer into JSON and BIN chunks.
 *
 * @param {ArrayBuffer} buffer GLB file bytes.
 * @returns {{ json: object, bin: Uint8Array }} Parsed chunks.
 */
export function parseGlbChunks(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 20 || u32(view, 0) !== GLB_MAGIC) {
    throw new Error("Not a GLB file.");
  }
  let offset = 12;
  /** @type {object | null} */
  let json = null;
  /** @type {Uint8Array} */
  let bin = new Uint8Array(0);

  while (offset + 8 <= buffer.byteLength) {
    const chunkLength = u32(view, offset);
    const chunkType = u32(view, offset + 4);
    const start = offset + 8;
    const end = start + chunkLength;
    if (end > buffer.byteLength) {
      break;
    }
    if (chunkType === JSON_CHUNK) {
      json = JSON.parse(new TextDecoder().decode(buffer.slice(start, end)));
    } else if (chunkType === BIN_CHUNK) {
      bin = new Uint8Array(buffer, start, chunkLength);
    }
    offset = end;
  }

  if (!json) {
    throw new Error("GLB JSON chunk is missing.");
  }
  return { json, bin };
}

/**
 * Collects world-space POSITION attributes from a glTF document.
 *
 * @param {object} json glTF JSON.
 * @param {Uint8Array} bin BIN bytes.
 * @returns {Float32Array} Interleaved xyz positions.
 */
export function collectWorldPositions(json, bin) {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const chunks = [];

  /**
   * @param {number} nodeIndex
   * @param {number[]} parent
   */
  const walk = (nodeIndex, parent) => {
    const node = json.nodes?.[nodeIndex];
    if (!node) {
      return;
    }
    const world = multiplyMat4(parent, nodeToMatrix(node));
    if (Number.isInteger(node.mesh) && json.meshes?.[node.mesh]) {
      for (const primitive of json.meshes[node.mesh].primitives || []) {
        const posIndex = primitive.attributes?.POSITION;
        if (!Number.isInteger(posIndex)) {
          continue;
        }
        chunks.push(transformPositions(readAccessorVec3(json, bin, posIndex), world));
      }
    }
    for (const child of node.children || []) {
      walk(child, world);
    }
  };

  const scene = json.scenes?.[json.scene ?? 0];
  const roots = scene?.nodes?.length ? scene.nodes : (json.nodes || []).map((_, i) => i);
  for (const root of roots) {
    walk(root, identity);
  }

  if (chunks.length === 0 && Array.isArray(json.meshes)) {
    for (const mesh of json.meshes) {
      for (const primitive of mesh.primitives || []) {
        const posIndex = primitive.attributes?.POSITION;
        if (Number.isInteger(posIndex)) {
          chunks.push(readAccessorVec3(json, bin, posIndex));
        }
      }
    }
  }

  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

/**
 * Reads all mesh positions from a GLB buffer.
 *
 * @param {ArrayBuffer} buffer GLB bytes.
 * @returns {Float32Array} Interleaved xyz positions.
 */
export function readGlbPositions(buffer) {
  const { json, bin } = parseGlbChunks(buffer);
  return collectWorldPositions(json, bin);
}

/**
 * Returns the median of a sorted numeric array.
 *
 * @param {number[]} sorted Sorted values.
 * @returns {number} Median.
 */
export function medianSorted(sorted) {
  if (sorted.length === 0) {
    return 0;
  }
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

/**
 * Linear interpolation quantile of a sorted array.
 *
 * @param {number[]} sorted Sorted values.
 * @param {number} q Quantile in [0, 1].
 * @returns {number} Quantile value.
 */
export function quantileSorted(sorted, q) {
  if (sorted.length === 0) {
    return 0;
  }
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[index];
}

/**
 * Computes a robust orbit target and radius from a point cloud.
 *
 * Uses per-axis medians so COLMAP outliers do not pull the pivot off the object.
 *
 * @param {Float32Array | number[]} positions Interleaved xyz positions.
 * @returns {{ cameraTarget: string, cameraOrbit: string } | null} model-viewer camera strings.
 */
export function computeOrbitFraming(positions) {
  const count = Math.floor(positions.length / 3);
  if (count < 1) {
    return null;
  }

  const xs = [];
  const ys = [];
  const zs = [];
  for (let i = 0; i < count; i += 1) {
    xs.push(positions[i * 3]);
    ys.push(positions[i * 3 + 1]);
    zs.push(positions[i * 3 + 2]);
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);

  const cx = medianSorted(xs);
  const cy = medianSorted(ys);
  const cz = medianSorted(zs);

  const distances = [];
  for (let i = 0; i < count; i += 1) {
    distances.push(
      Math.hypot(positions[i * 3] - cx, positions[i * 3 + 1] - cy, positions[i * 3 + 2] - cz),
    );
  }
  distances.sort((a, b) => a - b);
  const radius = Math.max(quantileSorted(distances, 0.9) * 2.4, 1e-4);

  return {
    cameraTarget: `${cx}m ${cy}m ${cz}m`,
    cameraOrbit: `0deg 75deg ${radius}m`,
  };
}

/**
 * Applies orbit framing to a model-viewer element.
 *
 * @param {HTMLElement} viewerEl Viewer element.
 * @param {{ cameraTarget: string, cameraOrbit: string }} framing Camera strings.
 * @returns {void}
 */
export function applyOrbitFraming(viewerEl, framing) {
  viewerEl.setAttribute("camera-target", framing.cameraTarget);
  viewerEl.setAttribute("camera-orbit", framing.cameraOrbit);
  if ("cameraTarget" in viewerEl) {
    viewerEl.cameraTarget = framing.cameraTarget;
  }
  if ("cameraOrbit" in viewerEl) {
    viewerEl.cameraOrbit = framing.cameraOrbit;
  }
  if (typeof viewerEl.jumpCameraToGoal === "function") {
    viewerEl.jumpCameraToGoal();
  }
}

let frameRequestId = 0;

/**
 * Resolves when the viewer has loaded, or immediately for plain test stubs.
 *
 * @param {HTMLElement} viewerEl Viewer element.
 * @returns {Promise<void>}
 */
function whenViewerReady(viewerEl) {
  if (typeof viewerEl.loaded === "boolean") {
    if (viewerEl.loaded) {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      viewerEl.addEventListener("load", () => resolve(), { once: true });
    });
  }
  return Promise.resolve();
}

/**
 * Frames model-viewer so orbit / auto-rotate pivot on the dense part of a GLB cloud.
 *
 * @param {HTMLElement} viewerEl `<model-viewer>` element.
 * @param {string} glbUrl GLB URL already set as ``src``.
 * @param {typeof fetch} [fetchImpl] Fetch implementation.
 * @returns {Promise<boolean>} True when framing was applied.
 */
export async function frameViewerOnCloud(viewerEl, glbUrl, fetchImpl = fetch) {
  const requestId = (frameRequestId += 1);
  try {
    const response = await fetchImpl(glbUrl);
    if (!response.ok) {
      return false;
    }
    const buffer = await response.arrayBuffer();
    await whenViewerReady(viewerEl);
    if (requestId !== frameRequestId) {
      return false;
    }
    const framing = computeOrbitFraming(readGlbPositions(buffer));
    if (!framing || requestId !== frameRequestId) {
      return false;
    }
    applyOrbitFraming(viewerEl, framing);
    return true;
  } catch {
    return false;
  }
}
