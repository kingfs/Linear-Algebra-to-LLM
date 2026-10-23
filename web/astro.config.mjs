import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export default defineConfig({
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
