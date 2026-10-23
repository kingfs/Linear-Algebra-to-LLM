import { useEffect, useMemo, useState } from "react";
import { idbGetAll, type Attempt, type CardState, type Progress } from "../lib/db";
import { withBase } from "../lib/base";

type Module = { id: string; title: string; route: string; concepts: { id: string; title: string; route: string }[] };

export default function ProgressPanel({
  modules,
  quizbank,
  totalConcepts,
}: {
  modules: Module[];
  quizbank: Record<string, any[]>;
  totalConcepts: number;
}) {
  const [progress, setProgress] = useState<Progress[]>([]);
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [cards, setCards] = useState<CardState[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    Promise.all([idbGetAll<Progress>("progress"), idbGetAll<Attempt>("attempts"), idbGetAll<CardState>("cards")])
      .then(([p, a, c]) => { setProgress(p); setAttempts(a); setCards(c); })
      .catch(() => {})
      .finally(() => setLoaded(true));
  }, []);

  const readSet = useMemo(() => new Set(progress.filter((p) => p.readAt).map((p) => p.route)), [progress]);
  const totalAttempts = attempts.length;
  const correct = attempts.filter((a) => a.correct).length;
  const accuracy = totalAttempts ? Math.round((correct / totalAttempts) * 100) : 0;
  const now = Date.now();
  const due = cards.filter((c) => c.due <= now).length;
  const readCount = readSet.size;
  const conceptTitle = useMemo(() => {
    const map = new Map<string, { title: string; route: string }>();
    modules.forEach((m) => m.concepts.forEach((c) => map.set(c.id, c)));
    return map;
  }, [modules]);

  const recent = [...attempts].sort((a, b) => b.at - a.at).slice(0, 8);
  const days = new Set([...attempts.map((a) => new Date(a.at).toDateString()), ...progress.filter((p) => p.readAt).map((p) => new Date(p.readAt).toDateString())]);
  const weak = Object.entries(
    attempts.reduce<Record<string, { total: number; wrong: number }>>((acc, a) => {
      const x = (acc[a.concept] ||= { total: 0, wrong: 0 });
      x.total++;
      if (!a.correct) x.wrong++;
      return acc;
    }, {}),
  )
    .map(([concept, v]) => ({ concept, ...v, rate: v.wrong / v.total }))
    .filter((x) => x.wrong > 0)
    .sort((a, b) => b.rate - a.rate || b.wrong - a.wrong)
    .slice(0, 5);

  function exportData() {
    const blob = new Blob([JSON.stringify({ progress, attempts, cards, exportedAt: new Date().toISOString() }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `linear-algebra-progress-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  if (!loaded) return <div className="empty">正在读取本地学习记录…</div>;

  return (
    <div>
      <div className="stats" style={{ marginTop: 18 }}>
        <div className="stat"><b>{readCount} / {totalConcepts}</b><span>已读概念</span></div>
        <div className="stat"><b>{accuracy}%</b><span>答题正确率 · {totalAttempts} 次</span></div>
        <div className="stat"><b>{due}</b><span>今日到期卡片</span></div>
        <div className="stat"><b>{days.size}</b><span>有学习记录的天数</span></div>
      </div>

      <div className="section-head"><h2>模块进度</h2><button className="btn ghost" onClick={exportData}>导出学习数据</button></div>
      <div className="progress-grid">
        {modules.map((m) => {
          const done = m.concepts.filter((c) => readSet.has(c.route)).length;
          const q = m.concepts.reduce((s, c) => s + (quizbank[c.id]?.length || 0), 0);
          const pct = Math.round((done / Math.max(1, m.concepts.length)) * 100);
          return (
            <a className="module-progress" href={withBase(m.route)} key={m.id} style={{ color: "var(--ink)" }}>
              <h4>{m.id} · {m.title.replace(/^M\d\d\s*/, "")}</h4>
              <div className="pct">{done} / {m.concepts.length} 篇已读 · {q} 题</div>
              <div className="bar"><i style={{ width: `${pct}%` }} /></div>
            </a>
          );
        })}
      </div>

      <div className="section-head"><h2>最近作答</h2><a href={withBase("/practice/")}>去练习 →</a></div>
      {recent.length === 0 ? (
        <div className="empty">还没有答题记录。去<a href={withBase("/practice/")}>练习场</a>做几题，这里会出现正确率与薄弱点。</div>
      ) : (
        <div className="grid" style={{ gap: 8 }}>
          {recent.map((a) => (
            <div key={a.id} style={{ display: "flex", gap: 12, alignItems: "center", background: "var(--bg-elev)", border: "1px solid var(--line)", borderRadius: 10, padding: "10px 14px" }}>
              <span className={`badge ${a.correct ? "green" : "amber"}`}>{a.correct ? "✓ 正确" : "✗ 错误"}</span>
              <a href={withBase(conceptTitle.get(a.concept)?.route || "#")} style={{ color: "var(--ink)", fontWeight: 600 }}>
                {a.concept} {conceptTitle.get(a.concept)?.title || ""}
              </a>
              <span style={{ marginLeft: "auto", color: "var(--ink-faint)", fontSize: 12 }}>{new Date(a.at).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })}</span>
            </div>
          ))}
        </div>
      )}

      <div className="section-head"><h2>薄弱概念</h2><a href={withBase("/wrong/")}>打开错题本 →</a></div>
      {weak.length === 0 ? (
        <div className="empty">做几道题后，这里会按错误率列出需要回看的概念。</div>
      ) : (
        <div className="grid" style={{ gap: 8 }}>
          {weak.map((w) => (
            <a key={w.concept} href={withBase(conceptTitle.get(w.concept)?.route || "#")} className="module-card" style={{ padding: "12px 14px" }}>
              <div className="m-id">{w.concept} · 错 {w.wrong} / {w.total}</div>
              <h3>{conceptTitle.get(w.concept)?.title || ""}</h3>
              <div className="bar"><i style={{ width: `${Math.round(w.rate * 100)}%`, background: "var(--amber)" }} /></div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
