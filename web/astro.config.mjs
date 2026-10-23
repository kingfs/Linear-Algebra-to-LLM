import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const rawBase = process.env.BASE_PATH || "/";
const base = rawBase.endsWith("/") ? rawBase : rawBase + "/";

export default defineConfig({
  base,
  integrations: [react()],
  trailingSlash: "always",
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { strict: false, throwOnError: false, output: "html" }]],
    shikiConfig: { theme: "github-dark", wrap: true },
  },
  vite: {
    build: { chunkSizeWarningLimit: 1200 },
  },
});
