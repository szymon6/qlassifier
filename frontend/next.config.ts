import type { NextConfig } from "next";

// Static export served by `qlassifier ui` (TDD §10.1, ADR 0008). No server features.
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  devIndicators: false,
  turbopack: {
    // Pin the workspace root: a stray lockfile higher up must not change the build.
    root: __dirname,
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
