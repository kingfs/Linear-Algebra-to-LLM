import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const docs = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/generated" }),
  schema: z.object({
    id: z.string(),
    title: z.string(),
    module: z.string(),
    slug: z.string(),
    route: z.string(),
    kind: z.enum(["concept", "module", "page", "exercise"]),
    order: z.number().default(0),
    minutes: z.number().default(20),
    level: z.string().default("核心"),
    tags: z.array(z.string()).default([]),
    llm_hooks: z.array(z.string()).default([]),
    prereq: z.array(z.string()).default([]),
    source: z.string().default(""),
    moduleTitle: z.string().default(""),
  }),
});

export const collections = { docs };
