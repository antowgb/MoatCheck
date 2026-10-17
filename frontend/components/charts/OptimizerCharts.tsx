"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { QuantFrontierPoint } from "@/lib/api";

const COLORS = ["#22C55E", "#38bdf8", "#fbbf24", "#8B5CF6"];

export interface OptimizerWeight {
  name: string;
  weight: number;
}

export function FrontierScatterChart({
  frontier,
  maxSharpe,
  minimumVariance,
  recommendation,
  riskLimit,
}: {
  frontier: Array<QuantFrontierPoint & { returnPct: number }>;
  maxSharpe: QuantFrontierPoint | null;
  minimumVariance: QuantFrontierPoint | null;
  recommendation: QuantFrontierPoint | null;
  riskLimit: number;
}) {
  return (
    <ResponsiveContainer>
      <ScatterChart margin={{ top: 12, right: 18, bottom: 22, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.28} />
        <XAxis type="number" dataKey="volatility" name="Volatility" tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis type="number" dataKey="returnPct" name="Return" tickFormatter={(value: number) => `${value.toFixed(0)}%`} tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip formatter={(value: number | string, name: string) => [typeof value === "number" ? `${value.toFixed(2)}${name === "Return" ? "%" : ""}` : value, name]} labelFormatter={() => "Portfolio"} contentStyle={{ borderRadius: 10, border: "1px solid #334155", background: "#0f172a", color: "#e2e8f0", fontSize: 12 }} />
        <ReferenceLine x={riskLimit} stroke="#fb7185" strokeDasharray="5 5" label={{ value: "Risk cap", fill: "#fb7185", fontSize: 10 }} />
        <Scatter name="Efficient frontier" data={frontier} fill="#34d399" line={{ stroke: "#34d399", strokeWidth: 2 }} shape="circle" />
        {maxSharpe && <Scatter name="Unconstrained maximum Sharpe" data={[{ ...maxSharpe, returnPct: (maxSharpe.expected_return ?? 0) * 100 }]} fill="#fbbf24" shape="star" />}
        {minimumVariance && <Scatter name="Minimum volatility" data={[{ ...minimumVariance, returnPct: (minimumVariance.expected_return ?? 0) * 100 }]} fill="#a78bfa" shape="diamond" />}
        {recommendation && <Scatter name="Model recommendation" data={[{ ...recommendation, returnPct: (recommendation.expected_return ?? 0) * 100 }]} fill="#38bdf8" shape="star" />}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function AllocationPieChart({ data }: { data: OptimizerWeight[] }) {
  return (
    <ResponsiveContainer>
      <PieChart><Pie data={data} dataKey="weight" nameKey="name" innerRadius={55} outerRadius={82} paddingAngle={3} cornerRadius={7} stroke="#111720" strokeWidth={3}>{data.map((entry, index) => <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />)}</Pie><Tooltip formatter={(value: number) => `${(Number(value) * 100).toFixed(1)}%`} contentStyle={{ borderRadius: 14, background: "#17202C", border: "1px solid rgba(255,255,255,.08)", color: "white", fontSize: 12 }} /></PieChart>
    </ResponsiveContainer>
  );
}

export function AllocationBarChart({ data }: { data: OptimizerWeight[] }) {
  return (
    <ResponsiveContainer>
      <BarChart data={data} layout="vertical" margin={{ left: 10, right: 24, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#334155" opacity={0.25} />
        <XAxis type="number" domain={[0, 1]} tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`} tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="name" width={58} tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip formatter={(value: number) => `${(Number(value) * 100).toFixed(1)}%`} contentStyle={{ borderRadius: 10, background: "#0f172a", border: 0, color: "white", fontSize: 12 }} />
        <Bar dataKey="weight" name="Weight" radius={[10, 10, 10, 10]}>{data.map((item, index) => <Cell key={item.name} fill={COLORS[index % COLORS.length]} />)}</Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
