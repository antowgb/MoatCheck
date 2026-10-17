import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        obsidian: "#0B0F14",
        slateCard: "#111720",
        glass: "#17202C",
        emeraldBrand: "#22C55E",
        emeraldSoft: "#10B981",
        amethyst: "#8B5CF6",
        coral: "#FF5252",
      },
      borderRadius: {
        bento: "24px",
        "bento-xl": "28px",
      },
      boxShadow: {
        "neon-emerald": "0 0 22px rgba(34, 197, 94, 0.18)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.35s ease-out both",
      },
    },
  },
  plugins: [],
};

export default config;
