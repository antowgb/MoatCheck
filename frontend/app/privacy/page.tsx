import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import PrivacyDataControls from "./PrivacyDataControls";

export default function PrivacyPage() {
  return (
    <div className="w-full">
      <PageHeader
        title="Privacy & data handling"
        subtitle="What MoatCheck uses and saves when you use the site."
      />

      <div className="space-y-4">
        <Card>
          <h2 className="text-base font-semibold">Data used for calculations</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
            Calculators use the portfolio details you send to return results. If you save a portfolio, backtest, stress test, or Advisor recommendation, it is stored in MoatCheck’s database.
          </p>
          <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
            The Portfolio workspace also keeps its unsaved ticker and weight selection in this browser&apos;s local storage. Clear it from your browser settings if you want to remove that local copy.
          </p>
        </Card>

        <Card>
          <h2 className="text-base font-semibold">Account session</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
            Your sign-in is kept in this tab so it stays active when you refresh the page. The server checks your session and uses a secure cookie for API requests. Sign out to end your session. Never enter server keys on this site.
          </p>
        </Card>

        <Card>
          <h2 className="text-base font-semibold">Access controls and retention</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
            Saved records belong to your account. You can delete saved analysis below. Account deletion and data export are not available yet. Do not enter brokerage passwords or sensitive personal details.
          </p>
        </Card>

        <PrivacyDataControls />

        <Card>
          <h2 className="text-base font-semibold">Purpose</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
            MoatCheck is a public portfolio research project. It does not trade, sell your portfolio data, or promise investment results. It uses market data and public company information.
          </p>
        </Card>
      </div>
    </div>
  );
}
