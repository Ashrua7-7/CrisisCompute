import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const RESULTS_DIR = path.join(__dirname, '..', 'results')

export default defineConfig({
  plugins: [
    react(),
    // Serve ../results/*.json as /results/*  — no changes to existing project needed
    {
      name: 'results-file-server',
      configureServer(server) {
        server.middlewares.use('/results', (req, res, next) => {
          const safeName = path.basename(req.url || '')
          if (!safeName.endsWith('.json') && !safeName.endsWith('.png')) return next()
          const filePath = path.join(RESULTS_DIR, safeName)
          if (!fs.existsSync(filePath)) return next()
          const isJson = safeName.endsWith('.json')
          res.setHeader('Content-Type', isJson ? 'application/json' : 'image/png')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(fs.readFileSync(filePath))
        })
      },
    },
  ],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:7860',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
