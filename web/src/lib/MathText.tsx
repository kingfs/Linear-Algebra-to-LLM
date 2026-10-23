import katex from "katex";

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] as string));
}

function textHtml(s: string) {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

export default function MathText({ text, block = false }: { text: string; block?: boolean }) {
  if (!text) return null;
  const parts = String(text).split(/(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (!part) return null;
        if (part.startsWith("$$") && part.endsWith("$$")) {
          const tex = part.slice(2, -2);
          return (
            <span
              key={i}
              dangerouslySetInnerHTML={{ __html: katex.renderToString(tex, { displayMode: true, throwOnError: false, output: "html" }) }}
            />
          );
        }
        if (part.startsWith("$") && part.endsWith("$")) {
          const tex = part.slice(1, -1);
          return (
            <span
              key={i}
              dangerouslySetInnerHTML={{ __html: katex.renderToString(tex, { displayMode: block, throwOnError: false, output: "html" }) }}
            />
          );
        }
        return <span key={i} dangerouslySetInnerHTML={{ __html: textHtml(part) }} />;
      })}
    </>
  );
}
