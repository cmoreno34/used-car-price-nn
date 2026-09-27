import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base must match the repository name: GitHub Pages serves it under /used-car-price-nn/.
export default defineConfig({ plugins: [react()], base: "/used-car-price-nn/" });
