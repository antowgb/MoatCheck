export default function ScoreBadge({ score }: { score: number | null }) {
  if (score === null || score === undefined || !Number.isFinite(score)) {
    return <span className="font-mono text-xs text-slate-600">N/A</span>;
  }
  const color =
    score >= 70
      ? "border-emerald-300/15 bg-emerald-300/[0.08] text-emerald-200"
      : score >= 50
        ? "border-amber-300/15 bg-amber-300/[0.08] text-amber-200"
        : "border-rose-300/15 bg-rose-300/[0.08] text-rose-200";
  return (
    <span className={`inline-flex min-w-12 justify-center rounded-lg border px-2 py-1 text-[11px] font-mono font-semibold tabular-nums ${color}`}>
      {score.toFixed(1)}
    </span>
  );
}
