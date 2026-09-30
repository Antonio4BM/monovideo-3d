import { describe, expect, it, vi } from "vitest";
import {
  applyOrbitFraming,
  computeOrbitFraming,
  frameViewerOnCloud,
  medianSorted,
  nodeToMatrix,
  parseGlbChunks,
  quantileSorted,
  readGlbPositions,
  transformPositions,
} from "../glb-framing.js";

/**
 * Encodes a minimal GLB with a single POSITION accessor.
 *
 * @param {number[][]} points xyz triplets.
 * @param {{ translation?: number[] }} [node] Optional node transform.
 * @returns {ArrayBuffer} GLB bytes.
 */
function buildGlb(points, node = {}) {
  const count = points.length;
  const bin = new ArrayBuffer(count * 12);
  const floats = new Float32Array(bin);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  points.forEach((point, index) => {
    floats[index * 3] = point[0];
    floats[index * 3 + 1] = point[1];
    floats[index * 3 + 2] = point[2];
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], point[axis]);
      max[axis] = Math.max(max[axis], point[axis]);
    }
  });

  const json = {
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, ...node }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, mode: 0 }] }],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count,
        type: "VEC3",
        min,
        max,
      },
    ],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: bin.byteLength }],
    buffers: [{ byteLength: bin.byteLength }],
  };

  const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  const paddedJson = new Uint8Array(jsonBytes.length + jsonPad);
  paddedJson.set(jsonBytes);
  paddedJson.fill(0x20, jsonBytes.length);

  const binPad = (4 - (bin.byteLength % 4)) % 4;
  const binChunkLength = bin.byteLength + binPad;
  const total = 12 + 8 + paddedJson.length + 8 + binChunkLength;
  const out = new ArrayBuffer(total);
  const view = new DataView(out);
  const bytes = new Uint8Array(out);

  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, paddedJson.length, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytes.set(paddedJson, 20);

  const binHeader = 20 + paddedJson.length;
  view.setUint32(binHeader, binChunkLength, true);
  view.setUint32(binHeader + 4, 0x004e4942, true);
  bytes.set(new Uint8Array(bin), binHeader + 8);
  return out;
}

describe("medianSorted", () => {
  it("returns the middle value for an odd-length list", () => {
    expect(medianSorted([1, 2, 9])).toBe(2);
  });

  it("averages the two middle values for an even-length list", () => {
    expect(medianSorted([1, 2, 3, 4])).toBe(2.5);
  });
});

describe("quantileSorted", () => {
  it("clamps to the last value", () => {
    expect(quantileSorted([1, 2, 3], 1)).toBe(3);
  });
});

describe("computeOrbitFraming", () => {
  it("uses the per-axis median so outliers do not move the pivot", () => {
    const positions = new Float32Array([
      1, 2, 3, 1.1, 2.1, 3.1, 0.9, 1.9, 2.9, 100, 100, 100,
    ]);
    const framing = computeOrbitFraming(positions);
    expect(framing).not.toBeNull();
    const [x, y, z] = framing.cameraTarget.replaceAll("m", "").split(" ").map(Number);
    expect(x).toBeCloseTo(1.05, 5);
    expect(y).toBeCloseTo(2.05, 5);
    expect(z).toBeCloseTo(3.05, 5);
    expect(framing.cameraOrbit).toMatch(/^0deg 75deg /);
  });

  it("returns null when there are no points", () => {
    expect(computeOrbitFraming(new Float32Array(0))).toBeNull();
  });
});

describe("readGlbPositions", () => {
  it("reads positions and applies node translation", () => {
    const buffer = buildGlb(
      [
        [0, 0, 0],
        [2, 0, 0],
      ],
      { translation: [10, 0, 0] },
    );
    const positions = readGlbPositions(buffer);
    expect(Array.from(positions)).toEqual([10, 0, 0, 12, 0, 0]);
  });

  it("parses JSON and BIN chunks", () => {
    const { json, bin } = parseGlbChunks(buildGlb([[1, 2, 3]]));
    expect(json.accessors[0].count).toBe(1);
    expect(bin.byteLength).toBeGreaterThanOrEqual(12);
  });

  it("rejects a non-GLB buffer", () => {
    expect(() => parseGlbChunks(new ArrayBuffer(8))).toThrow("Not a GLB file.");
  });
});

describe("nodeToMatrix", () => {
  it("returns an identity-like translation matrix", () => {
    const matrix = nodeToMatrix({ translation: [1, 2, 3] });
    expect(matrix[12]).toBe(1);
    expect(matrix[13]).toBe(2);
    expect(matrix[14]).toBe(3);
    expect(matrix[0]).toBe(1);
  });

  it("uses a provided matrix", () => {
    const matrix = Array.from({ length: 16 }, (_, i) => i);
    expect(nodeToMatrix({ matrix })).toEqual(matrix);
  });
});

describe("transformPositions", () => {
  it("applies translation from a matrix", () => {
    const matrix = nodeToMatrix({ translation: [1, 0, 0] });
    const out = transformPositions(new Float32Array([0, 0, 0]), matrix);
    expect(Array.from(out)).toEqual([1, 0, 0]);
  });
});

describe("applyOrbitFraming", () => {
  it("sets camera-target and camera-orbit attributes", () => {
    const viewerEl = document.createElement("div");
    applyOrbitFraming(viewerEl, {
      cameraTarget: "1m 2m 3m",
      cameraOrbit: "0deg 75deg 4m",
    });
    expect(viewerEl.getAttribute("camera-target")).toBe("1m 2m 3m");
    expect(viewerEl.getAttribute("camera-orbit")).toBe("0deg 75deg 4m");
  });

  it("jumps the camera when the viewer exposes jumpCameraToGoal", () => {
    const viewerEl = document.createElement("div");
    viewerEl.jumpCameraToGoal = vi.fn();
    applyOrbitFraming(viewerEl, {
      cameraTarget: "0m 0m 0m",
      cameraOrbit: "0deg 75deg 1m",
    });
    expect(viewerEl.jumpCameraToGoal).toHaveBeenCalled();
  });
});

describe("frameViewerOnCloud", () => {
  it("frames the viewer from a fetched GLB", async () => {
    const buffer = buildGlb([
      [1, 2, 3],
      [1, 2, 3],
    ]);
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      arrayBuffer: async () => buffer,
    }));
    const viewerEl = document.createElement("div");

    await expect(frameViewerOnCloud(viewerEl, "/model.glb", fetchImpl)).resolves.toBe(true);
    expect(viewerEl.getAttribute("camera-target")).toBe("1m 2m 3m");
  });

  it("returns false when fetch is not ok", async () => {
    const fetchImpl = vi.fn(async () => ({ ok: false }));
    const viewerEl = document.createElement("div");
    await expect(frameViewerOnCloud(viewerEl, "/missing.glb", fetchImpl)).resolves.toBe(false);
  });

  it("waits for the load event when loaded is false", async () => {
    const buffer = buildGlb([[5, 5, 5]]);
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      arrayBuffer: async () => buffer,
    }));
    const viewerEl = document.createElement("div");
    viewerEl.loaded = false;
    const addSpy = vi.spyOn(viewerEl, "addEventListener");
    const pending = frameViewerOnCloud(viewerEl, "/model.glb", fetchImpl);
    await vi.waitFor(() => {
      expect(addSpy).toHaveBeenCalledWith("load", expect.any(Function), { once: true });
    });
    viewerEl.dispatchEvent(new Event("load"));
    await expect(pending).resolves.toBe(true);
    expect(viewerEl.getAttribute("camera-target")).toBe("5m 5m 5m");
  });
});
