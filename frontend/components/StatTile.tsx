import MetricInfoTooltip from "@/components/explainability/MetricInfoTooltip";

export default function StatTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
}) {
  return (
    <div className="bento-panel min-w-0 rounded-2xl px-4 py-3.5 transition duration-200 hover:border-white/[0.1]">
      <p className="flex items-center text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">{label}<MetricInfoTooltip label={label} description={hint ?? "Saved portfolio data. Weights are fractions of the portfolio, expressed as percentages."} /></p>
      <p className="mt-1.5 truncate font-mono text-2xl font-semibold tabular-nums text-slate-100">{value}</p>
      {hint && <p className="mt-1 truncate text-[11px] text-slate-500">{hint}</p>}
    </div>
  );
}
