import type { Metadata } from "next";
import AppShell from "@/components/AppShell";
import { PortfolioProvider } from "@/components/PortfolioProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "MoatCheck | Portfolio Research", template: "%s | MoatCheck" },
  description: "Evidence-led quantitative portfolio research, risk analysis, and long-horizon advisory.",
  icons: { icon: "/logo.svg", shortcut: "/logo.svg", apple: "/logo.svg" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="font-sans">
        <PortfolioProvider>
          <AppShell>{children}</AppShell>
        </PortfolioProvider>
      </body>
    </html>
  );
}
