import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        soc: {
          bg: "#0a0d12",
          panel: "#10141b",
          raised: "#151a23",
          line: "#1e2530",
          lineStrong: "#2a3341",
          text: "#d9dfe8",
          muted: "#8a94a6",
          faint: "#5b6475",
          accent: "#38bdf8",
          accentDim: "#0c4a6e",
          danger: "#f43f5e",
          warn: "#f59e0b",
          ok: "#22c55e",
          violet: "#a78bfa",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Cascadia Mono", "Consolas", "monospace"],
      },
      keyframes: {
        flash: {
          "0%": { backgroundColor: "rgba(56,189,248,0.18)" },
          "100%": { backgroundColor: "transparent" },
        },
        flashDanger: {
          "0%": { backgroundColor: "rgba(244,63,94,0.28)" },
          "100%": { backgroundColor: "transparent" },
        },
        pulseDot: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.35", transform: "scale(0.8)" },
        },
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        flash: "flash 1.2s ease-out",
        flashDanger: "flashDanger 1.6s ease-out",
        pulseDot: "pulseDot 1.6s ease-in-out infinite",
        fadeIn: "fadeIn 0.25s ease-out",
      },
    },
  },
  plugins: [],
};

export default config;
