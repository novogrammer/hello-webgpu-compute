import { defineConfig } from 'vite';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: 'src',
  publicDir: '../public',
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'src/index.html'),
        wgsl_A: resolve(__dirname, 'src/wgsl/A_vector1d.html'),
        wgsl_Bn: resolve(__dirname, 'src/wgsl/B_blur_naive.html'),
        // wgsl_Bs: resolve(__dirname, 'src/wgsl/B_blur_shared.html'),
        // wgsl_C: resolve(__dirname, 'src/wgsl/C_game_of_life.html'),
        // tsl_A: resolve(__dirname, 'src/tsl/A_vector1d.html'),
        // tsl_Bn: resolve(__dirname, 'src/tsl/B_blur_naive.html'),
        // tsl_Bs: resolve(__dirname, 'src/tsl/B_blur_shared.html'),
        // tsl_C: resolve(__dirname, 'src/tsl/C_game_of_life.html')
      }
    }
  }
});