/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#0d1117',
          card: '#161b22',
          hover: '#1c2333',
          border: '#30363d',
          text: '#e6edf3',
          muted: '#8b949e',
        },
        accent: {
          blue: '#58a6ff',
          green: '#3fb950',
          red: '#f85149',
          yellow: '#d29922',
          purple: '#bc8cff',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
