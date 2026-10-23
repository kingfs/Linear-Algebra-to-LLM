import { useEffect, useState } from "react";
import { idbGet, idbPut, type Progress } from "../lib/db";

export default function MarkRead({ route }: { route: string }) {
  const [read, setRead] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    idbGet<Progress>("progress", route)
      .then((p) => setRead(Boolean(p?.readAt)))
      .catch(() => {})
      .finally(() => setReady(true));
  }, [route]);

  async function toggle() {
    if (read) {
      setRead(false);
      try { await idbPut("progress", { route, readAt: 0 }); } catch {}
    } else {
      setRead(true);
      try { await idbPut("progress", { route, readAt: Date.now() }); } catch {}
    }
  }

  return (
    <button className={`btn ${read ? "" : "primary"}`} onClick={toggle} disabled={!ready}>
      {read ? "✓ 已读" : "标记为已读"}
    </button>
  );
}
