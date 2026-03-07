import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // React core
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          // Firebase
          'vendor-firebase': ['firebase/app', 'firebase/auth', 'firebase/firestore'],
          // Charts
          'vendor-charts': ['recharts'],
          // Icons
          'vendor-icons': ['lucide-react'],
          // Utilities
          'vendor-utils': ['date-fns'],
        },
      },
    },
  },
})
