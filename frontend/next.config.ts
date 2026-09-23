import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: process.env.DEEP_RAG_DESKTOP_EXPORT === "1" ? "export" : "standalone",
  poweredByHeader: false,
};

export default nextConfig;
