import { useEffect, useMemo, useState } from "react";
import MathText from "../lib/MathText";
import { idbGetAll, idbPut, type CardState } from "../lib/db";
import { withBase } from "../lib/base";

type Card = { id: string; concept: string; title: string; route: string; front: string; back: string };
const DAY = 86_400_000;
const INTERVALS = [1, 3, 7, 14, 30, 60];

function schedule(prev: CardState | undefined, rating: 0 | 1 | 2 | 3): CardState {
  const now = Date.now();
  let box = prev?.box ?? 0;
  let due = now + DAY;
  let lapses = prev?.lapses ?? 0;
  if (rating === 0) { box = 0; due = now + 10 * 60_000; lapses++; }
  else if (rating === 1) { box = Math.max(1, box); due = now + DAY; }
  else if (rating === 2) { box = Math.min(INTERVALS.length, box + 1); due = now + INTERVALS[Math.min(box - 1, INTERVALS.length - 1)] * DAY; }
  else { box = Math.min(INTERVALS.length, box + 2); due = now + INTERVALS[Math.min(box - 1, INTERVALS.length - 1)] * DAY; }
  return { id: prev?.id || "", box, due, seen: (prev?.seen ?? 0) + 1, lapses };
}

export default function ReviewDeck({ cards }: { cards: Card[] }) {
  const [states, setStates] = useState<Record<string, CardState>>({});
  const [loaded, setLoaded] = useState(false);
  const [queue, setQueue] = useState<Card[]>([]);
  const [flipped, setFlipped] = useState(false);
  const [reviewed, setReviewed] = useState(0);

  useEffect(() => {
    idbGetAll<CardState>("cards")
      .then((rows) => {
        const map: Record<string, CardState> = {};
        rows.forEach((r) => (map[r.id] = r));
        setStates(map);
        const now = Date.now();
        const due = cards.filter((c) => !map[c.id] || map[c.id].due <= now);
        setQueue(due.sort((a, b) => (map[a.id]?.due || 0) - (map[b.id]?.due || 0)));
      })
      .catch(() => setQueue(cards))
      .finally(() => setLoaded(true));
  }, [cards]);

  const current = queue[0];
  const nextDue = useMemo(() => {
    if (!Object.keys(states).length) return null;
    const times = cards.map((c) => states[c.id]?.due).filter((t): t is number => Boolean(t));
    if (!times.length) return null;
    const min = Math.min(...times);
    return min > Date.now() ? new Date(min) : null;
  }, [cards, states]);

  async function grade(rating: 0 | 1 | 2 | 3) {
    if (!current) return;
    const next = schedule(states[current.id], rating);
    next.id = current.id;
    setStates((s) => ({ ...s, [current.id]: next }));
    setQueue((q) => q.slice(1));
    setFlipped(false);
    setReviewed((n) => n + 1);
    try { await idbPut("cards", next); } catch {}
  }

  if (!loaded) return <div className="empty">正在读取知识卡…</div>;

  const total = cards.length;
  const dueToday = queue.length;

  return (
    <div>
      <div className="review-stats">
        <span className="badge accent">今日到期 {dueToday}</span>
        <span className="badge">本次已复习 {reviewed}</span>
        <span className="badge">卡片总数 {total}</span>
      </div>

      {!current ? (
        <div className="empty" style={{ padding: 28 }}>
          <strong>今天的卡片复习完了。</strong>
          <p style={{ margin: "8px 0 0" }}>
            {nextDue ? `下一张在 ${nextDue.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })} 到期。` : "去读几篇概念，或做几道题，新的卡片会从这里长出来。"}
          </p>
          <div style={{ marginTop: 14, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <a className="btn primary" href={withBase("/practice/")}>去做题</a>
            <a className="btn" href={withBase("/")}>回首页</a>
          </div>
        </div>
      ) : (
        <>
          <div className={`flashcard ${flipped ? "flipped" : ""}`} onClick={() => setFlipped((f) => !f)}>
            <div className="flashcard-inner">
              <div className="flash-face">
                <div className="k">{current.concept} · {current.title}</div>
                <div className="v"><MathText text={current.front} /></div>
                <div style={{ color: "var(--ink-faint)", fontSize: 12 }}>点击翻面</div>
              </div>
              <div className="flash-face back">
                <div className="k">答案</div>
                <div className="v"><MathText text={current.back} /></div>
                <div style={{ color: "var(--ink-faint)", fontSize: 12 }}>根据回忆质量选择</div>
              </div>
            </div>
          </div>
          <div className="card-controls">
            <button className="btn" onClick={() => grade(0)} disabled={!flipped}>重来 <span style={{ color: "var(--ink-faint)", fontWeight: 400 }}>10 分钟</span></button>
            <button className="btn" onClick={() => grade(1)} disabled={!flipped}>困难 <span style={{ color: "var(--ink-faint)", fontWeight: 400 }}>1 天</span></button>
            <button className="btn primary" onClick={() => grade(2)} disabled={!flipped}>良好 <span style={{ opacity: .75, fontWeight: 400 }}>按间隔</span></button>
            <button className="btn" onClick={() => grade(3)} disabled={!flipped}>简单 <span style={{ color: "var(--ink-faint)", fontWeight: 400 }}>跳级</span></button>
            <a className="btn ghost" href={withBase(current.route)}>回原文</a>
          </div>
          <p style={{ color: "var(--ink-faint)", fontSize: 12, marginTop: 10 }}>
            卡片由每篇「一页速记」自动生成；评分只保存在本机 IndexedDB，不上传。
          </p>
        </>
      )}
    </div>
  );
}
