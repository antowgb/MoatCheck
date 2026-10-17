import type { SVGProps } from "react";

type LogoProps = SVGProps<SVGSVGElement> & {
  size?: number;
};

/** Brand mark matching `/public/logo.svg`; scales cleanly in headers and nav rails. */
export default function Logo({ size = 40, className = "", ...props }: LogoProps) {
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      className={`logo-glow shrink-0 ${className}`}
      {...props}
    >
      <path d="M10 52 25 38 35 43 54 12" stroke="#22C55E" strokeWidth="7.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
