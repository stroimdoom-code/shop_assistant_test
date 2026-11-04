import { defineConfig } from "vite";

// Браузер работает с origin Vite, API остаётся на локальном FastAPI.
export default defineConfig({
  server: { proxy: { "/api": { target: "http://127.0.0.1:8000" } } },
  build: { outDir: "dist", emptyOutDir: true },
});
