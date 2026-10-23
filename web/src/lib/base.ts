/**
 * 部署在子路径时（例如 /Linear-Algebra-to-LLM/），所有站内链接都要带上 BASE_URL。
 * 本地开发默认 BASE_URL=/，行为不变。
 */
export const BASE = import.meta.env.BASE_URL || "/";

const BASE_PREFIX = BASE.endsWith("/") ? BASE : BASE + "/";

export function withBase(path: string): string {
  if (!path) return BASE_PREFIX;
  if (/^(?:[a-z]+:)?\/\//i.test(path) || path.startsWith("#")) return path;
  if (/^(?:mailto|tel|javascript):/i.test(path)) return path;
  if (BASE_PREFIX !== "/" && path.startsWith(BASE_PREFIX)) return path;
  if (path === "/") return BASE_PREFIX;
  return BASE_PREFIX + path.replace(/^\/+/, "");
}
