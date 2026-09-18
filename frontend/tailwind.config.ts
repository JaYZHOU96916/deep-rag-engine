import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#11233A",
        paper: "#FBFCFD",
        fog: "#E8EEF2",
        signal: "#3368D6",
        verify: "#005E6A",
        marker: "#D4A72C",
      },
      boxShadow: {
        ledger: "0 18px 44px rgba(17, 35, 58, 0.08)",
      },
    },
  },
  plugins: [],
};

export default config;
