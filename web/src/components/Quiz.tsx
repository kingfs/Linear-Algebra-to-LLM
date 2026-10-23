import { useMemo, useState } from "react";
import MathText from "../lib/MathText";
import { idbPut, type Attempt } from "../lib/db";
import { withBase } from "../lib/base";

export type Question = {
  id: string;
  concept: string;
  type: "single_choice" | "multiple_choice" | "true_false" | "numeric";
  difficulty?: number;
  stem: string;
  options?: string[];
  answer: number[] | number | boolean;
  tolerance?: number;
  unit?: string;
  explanation: string;
  source?: string;
  sourceRoute?: string;
};

const LETTERS = ["A", "B", "C", "D", "E", "F"];

function correctIndexes(q: Question): number[] {
  if (q.type === "true_false") {
    if (Array.isArray(q.answer)) return q.answer;
    return [q.answer === true ? 0 : 1];
  }
  if (q.type === "numeric") return [];
  return Array.isArray(q.answer) ? q.answer : [q.answer as number];
}

function optionsOf(q: Question): string[] {
  if (q.type === "true_false") return ["正确", "错误"];
  return q.options || [];
}

function isCorrect(q: Question, picked: number[], numeric: string): boolean {
  if (q.type === "numeric") {
    const v = Number(numeric);
    if (!Number.isFinite(v)) return false;
    const tol = q.tolerance ?? 1e-6;
    return Math.abs(v - Number(q.answer)) <= tol;
  }
  const correct = correctIndexes(q);
  const a = [...picked].sort((x, y) => x - y);
  const b = [...correct].sort((x, y) => x - y);
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

function answerText(q: Question): string {
  if (q.type === "numeric") return `${q.answer}${q.unit ? " " + q.unit : ""}`;
  if (q.type === "true_false") return q.answer === true || correctIndexes(q)[0] === 0 ? "正确" : "错误";
  return correctIndexes(q).map((i) => LETTERS[i]).join("、");
}

export default function Quiz({
  questions,
  concept,
  compact = false,
}: {
  questions: Question[];
  concept?: string;
  compact?: boolean;
}) {
  const [i, setI] = useState(0);
  const [picked, setPicked] = useState<number[]>([]);
  const [numeric, setNumeric] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [correct, setCorrect] = useState(false);
  const [results, setResults] = useState<boolean[]>([]);
  const [done, setDone] = useState(false);

  const q = questions[i];
  const options = useMemo(() => (q ? optionsOf(q) : []), [q]);
  if (!q) return <div className="empty">暂无题目。</div>;

  const isMulti = q.type === "multiple_choice";
  const canSubmit = q.type === "numeric" ? numeric.trim() !== "" : picked.length > 0;

  function toggle(idx: number) {
    if (submitted) return;
    if (isMulti) setPicked((p) => (p.includes(idx) ? p.filter((x) => x !== idx) : [...p, idx]));
    else setPicked([idx]);
  }

  async function submit() {
    if (!canSubmit || submitted) return;
    const ok = isCorrect(q, picked, numeric);
    setCorrect(ok);
    setSubmitted(true);
    setResults((r) => {
      const next = [...r];
      next[i] = ok;
      return next;
    });
    const attempt: Attempt = {
      id: `${q.id}-${Date.now()}`,
      questionId: q.id,
      concept: q.concept || concept || "",
      correct: ok,
      at: Date.now(),
    };
    try { await idbPut("attempts", attempt); } catch {}
  }

  function next() {
    if (i + 1 >= questions.length) {
      setDone(true);
      return;
    }
    setI(i + 1);
    setPicked([]);
    setNumeric("");
    setSubmitted(false);
    setCorrect(false);
  }

  function retry() {
    setI(0);
    setPicked([]);
    setNumeric("");
    setSubmitted(false);
    setCorrect(false);
    setResults([]);
    setDone(false);
  }

  if (done) {
    const score = results.filter(Boolean).length;
    const pct = Math.round((score / questions.length) * 100);
    return (
      <div className={`quiz ${compact ? "compact" : ""}`}>
        <div className="quiz-result">
          <div className="score">{score} / {questions.length}</div>
          <p>正确率 {pct}%{pct === 100 ? " · 全部答对，漂亮。" : pct >= 60 ? " · 基础过关，错题值得再看一遍。" : " · 建议回读相关小节，再做一次。"}</p>
          <div className="quiz-actions" style={{ justifyContent: "center" }}>
            <button className="btn primary" onClick={retry}>再做一遍</button>
            <a className="btn" href={q.sourceRoute ? withBase(q.sourceRoute) : "#"}>回看原文</a>
            <a className="btn ghost" href={withBase("/progress/")}>看学习进度</a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="quiz">
      <div className="quiz-top">
        <span className="badge accent">{q.type === "single_choice" ? "单选" : q.type === "multiple_choice" ? "多选" : q.type === "true_false" ? "判断" : "数值"}</span>
        {q.difficulty ? <span className="badge">难度 {q.difficulty}</span> : null}
        <span className="qcount">第 {i + 1} / {questions.length} 题</span>
      </div>

      <div className="q-stem"><MathText text={q.stem} /></div>

      {q.type === "numeric" ? (
        <div className="numeric-row">
          <input
            type="number"
            step="any"
            value={numeric}
            disabled={submitted}
            onChange={(e) => setNumeric(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
            placeholder="输入数值"
            aria-label="数值答案"
          />
          {q.unit && <span className="badge">{q.unit}</span>}
        </div>
      ) : (
        <div className="options" role={isMulti ? "group" : "radiogroup"}>
          {options.map((opt, idx) => {
            const selected = picked.includes(idx);
            const isAnswer = correctIndexes(q).includes(idx);
            const cls = [
              "option",
              selected ? "selected" : "",
              submitted && isAnswer ? "correct" : "",
              submitted && selected && !isAnswer ? "wrong" : "",
            ].filter(Boolean).join(" ");
            return (
              <label className={cls} key={idx}>
                <input
                  type={isMulti ? "checkbox" : "radio"}
                  name={`q-${q.id}`}
                  checked={selected}
                  disabled={submitted}
                  onChange={() => toggle(idx)}
                />
                <span className="key">{LETTERS[idx]}</span>
                <span><MathText text={opt} /></span>
              </label>
            );
          })}
        </div>
      )}

      {submitted && (
        <div className={`explain ${correct ? "good" : "bad"}`} aria-live="polite">
          <strong>{correct ? "✓ 正确" : "✗ 不对"}</strong>
          {!correct && <> · 正确答案：{answerText(q)}</>}
          <div style={{ marginTop: 6 }}><MathText text={q.explanation} /></div>
          {q.sourceRoute && <div style={{ marginTop: 8 }}><a href={withBase(q.sourceRoute)}>回看对应小节 →</a></div>}
        </div>
      )}

      <div className="quiz-actions">
        {!submitted ? (
          <button className="btn primary" onClick={submit} disabled={!canSubmit}>提交答案</button>
        ) : (
          <button className="btn primary" onClick={next}>{i + 1 >= questions.length ? "查看结果" : "下一题 →"}</button>
        )}
        <a className="btn ghost" href={q.sourceRoute ? withBase(q.sourceRoute) : "#"}>没思路？看原文</a>
      </div>
    </div>
  );
}
