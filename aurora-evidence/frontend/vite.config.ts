import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "");
  return {
    plugins: [react()],
    server: {
      port: Number(env.AURORA_FRONTEND_PORT || 5172),
      strictPort: true,
      proxy: {
        "/api":
          env.AURORA_API_INTERNAL_URL ||
          `http://127.0.0.1:${env.AURORA_PORT || 8102}`,
        "/health":
          env.AURORA_API_INTERNAL_URL ||
          `http://127.0.0.1:${env.AURORA_PORT || 8102}`,
        "/ready":
          env.AURORA_API_INTERNAL_URL ||
          `http://127.0.0.1:${env.AURORA_PORT || 8102}`,
      },
    },
  };
});
