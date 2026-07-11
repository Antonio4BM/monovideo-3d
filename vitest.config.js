import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["static/tests/**/*.test.js"],
    coverage: {
      provider: "v8",
      reportsDirectory: "static/tests/coverage",
      include: ["static/uploader.js"],
      thresholds: {
        lines: 75,
        functions: 75,
        branches: 75,
        statements: 75,
      },
    },
  },
});
