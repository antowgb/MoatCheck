import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Keep Next's workspace detection scoped to this app; the parent Projects
  // folder may contain unrelated package lockfiles.
  outputFileTracingRoot: process.cwd(),
  // Static export: the build produces `out/`, served by FastAPI in prod.
  output: "export",
  // The static export is served by FastAPI, which adds CSP and the full
  // security header set. Disable Next's framework identification header too.
  poweredByHeader: false,
  // Generates screener/index.html etc. so StaticFiles(html=True) resolves /screener/
  trailingSlash: true,
};

export default nextConfig;
