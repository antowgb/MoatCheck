export default function ChartFrame({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`h-[360px] w-full ${className}`}>{children}</div>;
}
