import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#f2efe9",
        ink: "#171611",
        accent: "#0f766e",
        ember: "#b45309",
      },
      boxShadow: {
        panel: "0 14px 40px rgba(23, 22, 17, 0.12)",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        rise: "rise 420ms ease-out both",
      },
    },
  },
  plugins: [],
};

export default config;
