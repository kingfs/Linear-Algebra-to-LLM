import { useEffect, useMemo, useState } from "react";
import Quiz, { type Question } from "./Quiz";
import MathText from "../lib/MathText";
import { idbGetAll, type Attempt } from "../lib/db";

export default function WrongBook({ questions }: { questions: Question[] }) {
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [deck, setDeck] = useState<Question[] | null>(null);

  useEffect(() => {
    idbGetAll<Attempt>("attempts").then(setAttempts).catch(() => {}).finally(() => setLoaded(true));
  }, []);

  const byId = useMemo(() => new Map(questions.map((q) => [q.id, q])), [questions]);
  const wrong = useMemo(() => {
    const latest = new Map<string, Attempt>();
    for (const a of attempts) {
      const prev = latest.get(a.questionId);
      if (!prev || a.at > prev.at) latest.set(a.questionId, a);
    }
    return [...latest.values()]
      .filter((a) => !a.correct && byId.has(a.questionId))
      .sort((a, b) => b.at - a.at)
      .map((a) => ({ attempt: a, question: byId.get(a.questionId)! }));
  }, [attempts, byId]);

  if (!loaded) return <div className="empty">正在读取错题…</div>;
  if (!wrong.length) {
    return (
      <div className="empty" style={{ padding: 28 }}>
        <strong>还没有错题。</strong>
        <p style={{ margin: "8px 0 0" }}>做错的题会自动出现在这里；连续答对后仍会保留，方便回看。</p>
        <div style={{ marginTop: 14 }}><a className="btn primary" href="/practice/">去练习</a></div>
      </div>
    );
  }
  if (deck) {
    return (
      <div>
        <button className="btn" onClick={() => setDeck(null)}>← 返回错题列表</button>
        <Quiz questions={deck} />
      </div>
    );
  }

  return (
    <div>
      <div className="review-stats">
        <span className="badge amber">{wrong.length} 道待复盘</span>
        <button className="btn primary" onClick={() => setDeck(wrong.map((w) => w.question))}>重做这些题</button>
      </div>
      <div className="grid" style={{ gap: 10 }}>
        {wrong.map(({ attempt, question }) => (
          <div className="module-card" key={question.id}>
            <div className="m-id">{question.concept} · 最近答错 {new Date(attempt.at).toLocaleDateString("zh-CN")}</div>
            <div style={{ margin: "6px 0", fontSize: 14.5 }}><MathText text={question.stem} /></div>
            <div className="m-meta" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <a className="btn" href={question.sourceRoute || "#"}>回看原文</a>
              <button className="btn ghost" onClick={() => setDeck([question])}>只重做这道</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
