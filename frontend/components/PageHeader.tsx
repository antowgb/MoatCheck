export default function PageHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="mb-7 animate-fade-up sm:mb-8">
      <h1 className="text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl">{title}</h1>
      {subtitle && <p className="mt-2 text-base leading-relaxed text-slate-400 sm:text-lg">{subtitle}</p>}
    </div>
  );
}
